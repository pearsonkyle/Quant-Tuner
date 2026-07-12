// jlens-server: llama.cpp-based activation server for the Jacobian Lens.
//
// Adapted from igorbarshteyn/jlens-gguf (Apache-2.0) for quant-tuner: built
// against the repo's vendored llama.cpp (vendor/llama.cpp) via build.sh, with
// the serving llama.cpp commit stamped into /props (JLENS_LLAMA_COMMIT) so
// capture runs record exactly which build produced them. See NOTICE at the
// repo root.
//
// Serves a GGUF model with llama.cpp and exposes, over HTTP, the residual
// stream at every layer/position ("l_out-<il>" tensors) plus live
// interventions (steering / low-rank edits / replacement) applied mid-graph
// through the ggml scheduler eval callback. Pairs with the jlens_gguf Python
// bridge, which does the lens math (J_l transport, unembedding) in numpy.
//
// Endpoints:
//   GET  /health           -> {"status":"ok"}
//   GET  /props            -> model metadata
//   GET  /vocab            -> all token pieces + attrs (one-time, cacheable)
//   POST /tokenize         {content, add_special?, parse_special?}
//   POST /detokenize       {tokens}
//   POST /apply_template   {messages:[{role,content}], add_assistant?}
//   POST /jlens/forward    the workhorse; binary response (see README)
//
// /jlens/forward request JSON:
// {
//   "tokens": [int, ...],                  // required
//   "capture_layers": [int, ...] | null,   // default: all layers
//   "dtype": "f16" | "f32",                // activation payload dtype (default f16)
//   "interventions": [
//     {"layer": l, "pos_start": p0, "pos_end": p1,   // p1 = -1 -> unbounded
//      "mode": "add" | "set" | "lowrank",
//      "data": "<base64 of little-endian f32>",       // add/set: [d]; lowrank: A [d,k] then B [k,d], row-major
//      "k": int }                                     // lowrank only
//   ],
//   "n_predict": 0,
//   "sampling": {"greedy": true, "temp": 0.8, "top_k": 40, "top_p": 0.95, "seed": -1},
//   "logits_positions": [int, ...]         // absolute positions to return raw model logits for
// }
//
// /jlens/forward response: "JLNS" | u32 version | u32 header_len | header JSON | payload
// header JSON:
// {
//   "tokens": [...], "n_prompt": int, "n_gen": int,
//   "generated": [{"token": id, "piece": str}, ...],
//   "activations": [{"layer": l, "dtype": "f16", "shape": [n_pos, d], "offset": o, "nbytes": n}, ...],
//   "logits": [{"pos": p, "offset": o, "nbytes": n}, ...],   // f32 [n_vocab]
//   "timings": {...}
// }

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

// llama.cpp commit this binary was built against (stamped by build.sh)
#ifndef JLENS_LLAMA_COMMIT
#define JLENS_LLAMA_COMMIT "unknown"
#endif

using json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------
// base64
// ---------------------------------------------------------------------------

static const char B64_CHARS[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::vector<uint8_t> base64_decode(const std::string & in) {
    static int8_t lut[256];
    static bool init = false;
    if (!init) {
        memset(lut, -1, sizeof(lut));
        for (int i = 0; i < 64; i++) lut[(uint8_t) B64_CHARS[i]] = (int8_t) i;
        init = true;
    }
    std::vector<uint8_t> out;
    out.reserve(in.size() * 3 / 4);
    int val = 0, bits = 0;
    for (char c : in) {
        if (c == '=' || c == '\n' || c == '\r') continue;
        int8_t d = lut[(uint8_t) c];
        if (d < 0) continue;
        val = (val << 6) | d;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t) ((val >> bits) & 0xFF));
        }
    }
    return out;
}

static std::string base64_encode(const uint8_t * data, size_t len) {
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = data[i] << 16;
        if (i + 1 < len) v |= data[i + 1] << 8;
        if (i + 2 < len) v |= data[i + 2];
        out.push_back(B64_CHARS[(v >> 18) & 63]);
        out.push_back(B64_CHARS[(v >> 12) & 63]);
        out.push_back(i + 1 < len ? B64_CHARS[(v >> 6) & 63] : '=');
        out.push_back(i + 2 < len ? B64_CHARS[v & 63] : '=');
    }
    return out;
}

static bool is_valid_utf8(const std::string & s) {
    const uint8_t * p = (const uint8_t *) s.data();
    const uint8_t * end = p + s.size();
    while (p < end) {
        if (*p < 0x80) { p++; continue; }
        int n;
        if      ((*p & 0xE0) == 0xC0) n = 1;
        else if ((*p & 0xF0) == 0xE0) n = 2;
        else if ((*p & 0xF8) == 0xF0) n = 3;
        else return false;
        if (p + n >= end) return false;
        for (int i = 1; i <= n; i++) if ((p[i] & 0xC0) != 0x80) return false;
        p += n + 1;
    }
    return true;
}

// token pieces can be raw bytes (byte-fallback tokens); JSON requires UTF-8,
// so invalid pieces travel as {"b64": "<base64>"} instead of plain strings
static json piece_json(const std::string & piece) {
    if (is_valid_utf8(piece)) return piece;
    return json{{"b64", base64_encode((const uint8_t *) piece.data(), piece.size())}};
}

static std::string dump_json(const json & j) {
    return j.dump(-1, ' ', false, nlohmann::detail::error_handler_t::replace);
}

// largest cut point <= end that does not split a UTF-8 sequence
static size_t utf8_safe_cut(const std::string & s, size_t end) {
    if (end > s.size()) end = s.size();
    size_t i = end;
    while (i > 0 && ((uint8_t) s[i - 1] & 0xC0) == 0x80) i--;   // continuation bytes
    if (i == 0) return end;
    const uint8_t lead = (uint8_t) s[i - 1];
    size_t need = 0;
    if      ((lead & 0xE0) == 0xC0) need = 2;
    else if ((lead & 0xF0) == 0xE0) need = 3;
    else if ((lead & 0xF8) == 0xF0) need = 4;
    else return end;                                             // ASCII or invalid: emit as-is
    return (end - (i - 1) >= need) ? end : i - 1;                // sequence complete? else cut before it
}

// incremental stop-string matcher with holdback, so a stop sequence split
// across token pieces is caught and never partially emitted
struct stop_scanner {
    std::vector<std::string> stops;
    size_t max_len = 0;
    std::string pending;
    bool hit = false;

    explicit stop_scanner(std::vector<std::string> s) : stops(std::move(s)) {
        // drop empty stop strings: "" matches everywhere and would terminate
        // every completion immediately (and leaves max_len == 0)
        stops.erase(std::remove(stops.begin(), stops.end(), std::string()), stops.end());
        for (const auto & x : stops) max_len = std::max(max_len, x.size());
    }

    // feed a piece; returns the text that is now safe to emit
    std::string feed(const std::string & piece) {
        if (hit) return "";
        pending += piece;
        for (const auto & s : stops) {
            const size_t p = pending.find(s);
            if (p != std::string::npos) {
                hit = true;
                std::string out = pending.substr(0, p);
                pending.clear();
                return out;
            }
        }
        size_t keep = 0;
        if (!stops.empty()) {
            // longest tail of `pending` that is a prefix of some stop string
            for (size_t k = std::min(pending.size(), max_len - 1); k > 0 && !keep; k--) {
                const std::string tail = pending.substr(pending.size() - k);
                for (const auto & s : stops) {
                    if (s.compare(0, k, tail) == 0) { keep = k; break; }
                }
            }
        }
        size_t cut = utf8_safe_cut(pending, pending.size() - keep);
        std::string out = pending.substr(0, cut);
        pending.erase(0, cut);
        return out;
    }

    std::string flush() {
        std::string out;
        out.swap(pending);
        return hit ? "" : out;
    }
};

// ---------------------------------------------------------------------------
// intervention + capture state shared with the eval callback
// ---------------------------------------------------------------------------

enum class iv_mode { ADD, SET, LOWRANK };

struct intervention {
    int     layer   = -1;
    int64_t p0      = 0;   // inclusive
    int64_t p1      = -1;  // exclusive; -1 = unbounded
    iv_mode mode    = iv_mode::ADD;
    int     k       = 0;                 // lowrank rank
    std::vector<float> a;                // add/set: [d]; lowrank: A [d,k] row-major
    std::vector<float> b;                // lowrank: B [k,d] row-major
};

struct cb_state {
    // configuration for the current request
    std::set<int>              capture_layers;   // empty = capture nothing
    std::vector<intervention>  interventions;
    int64_t                    d_model  = 0;
    int                        n_layer  = 0;

    // per-decode-call bookkeeping (set by the driver before llama_decode)
    int64_t base_pos = 0;   // absolute position of ubatch column 0
    int64_t expect_n = 0;   // expected ubatch width

    // capture accumulators: layer -> f32 data appended chunk by chunk
    std::map<int, std::vector<float>> captured;

    // set when a tensor looked wrong (non-contiguous / unexpected width)
    std::string error;

    void reset_request() {
        capture_layers.clear();
        interventions.clear();
        captured.clear();
        error.clear();
        base_pos = 0;
        expect_n = 0;
    }
};

static int parse_l_out(const char * name) {
    // accept "l_out-12" and backend-split-prefixed variants like "CUDA0#l_out-12#0"
    const char * p = strstr(name, "l_out-");
    if (!p) return -1;
    // guard against tensor names that merely contain the substring mid-word
    if (p != name && p[-1] != '#') return -1;
    char * end = nullptr;
    long il = strtol(p + 6, &end, 10);
    if (end == p + 6) return -1;
    if (*end != '\0' && *end != '#') return -1;
    return (int) il;
}

static bool jlens_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    cb_state * st = (cb_state *) user_data;
    const int il = parse_l_out(t->name);
    if (ask) {
        if (il < 0) return false;
        if (st->capture_layers.count(il)) return true;
        for (const auto & iv : st->interventions) {
            if (iv.layer == il) return true;
        }
        return false;
    }
    if (il < 0) return true;

    const int64_t d = t->ne[0];
    const int64_t n = t->ne[1];
    st->d_model = d;

    if (t->type != GGML_TYPE_F32) {
        st->error = std::string("l_out tensor is not f32: ") + ggml_type_name(t->type);
        return true;
    }
    if (t->nb[0] != sizeof(float) || (size_t) t->nb[1] != d * sizeof(float)) {
        st->error = "l_out tensor is not contiguous";
        return true;
    }
    if (st->expect_n > 0 && n != st->expect_n) {
        st->error = "unexpected ubatch width " + std::to_string(n) +
                    " (expected " + std::to_string(st->expect_n) + "); reduce --chunk?";
        return true;
    }

    // 1) interventions: modify the residual in place, column range by column range
    bool modified = false;
    std::vector<float> cols;
    for (const auto & iv : st->interventions) {
        if (iv.layer != il) continue;
        const int64_t hi = iv.p1 < 0 ? INT64_MAX : iv.p1;
        const int64_t s  = std::max<int64_t>(iv.p0, st->base_pos);
        const int64_t e  = std::min<int64_t>(hi, st->base_pos + n);
        if (s >= e) continue;
        const int64_t nc = e - s;              // columns to edit
        const int64_t c0 = s - st->base_pos;   // first local column
        cols.resize(nc * d);
        ggml_backend_tensor_get(t, cols.data(), c0 * d * sizeof(float), nc * d * sizeof(float));
        for (int64_t c = 0; c < nc; c++) {
            float * x = cols.data() + c * d;
            switch (iv.mode) {
                case iv_mode::ADD:
                    for (int64_t i = 0; i < d; i++) x[i] += iv.a[i];
                    break;
                case iv_mode::SET:
                    memcpy(x, iv.a.data(), d * sizeof(float));
                    break;
                case iv_mode::LOWRANK: {
                    // x += A (B x);  A: [d,k] row-major, B: [k,d] row-major
                    const int k = iv.k;
                    std::vector<float> tvec(k, 0.0f);
                    for (int j = 0; j < k; j++) {
                        const float * brow = iv.b.data() + (size_t) j * d;
                        double acc = 0.0;
                        for (int64_t i = 0; i < d; i++) acc += (double) brow[i] * x[i];
                        tvec[j] = (float) acc;
                    }
                    for (int64_t i = 0; i < d; i++) {
                        const float * arow = iv.a.data() + (size_t) i * k;
                        float acc = 0.0f;
                        for (int j = 0; j < k; j++) acc += arow[j] * tvec[j];
                        x[i] += acc;
                    }
                    break;
                }
            }
        }
        ggml_backend_tensor_set(t, cols.data(), c0 * d * sizeof(float), nc * d * sizeof(float));
        modified = true;
    }
    (void) modified;

    // 2) capture (post-intervention, i.e. the residual the model actually uses)
    if (st->capture_layers.count(il)) {
        auto & dst = st->captured[il];
        const size_t off = dst.size();
        dst.resize(off + (size_t) n * d);
        ggml_backend_tensor_get(t, dst.data() + off, 0, (size_t) n * d * sizeof(float));
    }
    return true;
}

// ---------------------------------------------------------------------------
// server context
// ---------------------------------------------------------------------------

struct jlens_server {
    llama_model *       model = nullptr;
    llama_context *     ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    cb_state            cb;
    std::mutex          mutex;   // serializes /jlens/forward (and anything touching ctx)

    std::string model_path;
    int n_layer  = 0;
    int n_embd   = 0;
    int n_vocab  = 0;
    int n_ctx    = 4096;
    int chunk    = 512;
    bool l_out_ok = false;
    std::string model_name;

    std::string vocab_json;  // cached /vocab response

    // ---- backend mode: server-held intervention set for /v1 completions ----
    std::vector<intervention> live_ivs;      // decoded, applied to every completion
    std::string live_iv_raw = "[]";          // canonical JSON of the set (echo + cache hash)
    json live_iv_meta;                       // opaque metadata (the bridge stores UI specs here)
    json last_completion;                    // {id, tokens, n_prompt, n_gen, text, finish_reason}
    uint64_t completion_counter = 0;
    std::vector<llama_token> cache_tokens;   // tokens whose KV is live in seq 0
    size_t cache_iv_hash = std::hash<std::string>{}("[]");

    std::string token_piece(llama_token tok, bool special) const {
        std::string piece(64, '\0');
        int n = llama_token_to_piece(vocab, tok, piece.data(), (int) piece.size(), 0, special);
        if (n < 0) {
            piece.resize(-n);
            n = llama_token_to_piece(vocab, tok, piece.data(), (int) piece.size(), 0, special);
        }
        piece.resize(std::max(n, 0));
        return piece;
    }
};

static std::vector<llama_token> tokenize_str(const jlens_server & S, const std::string & content,
                                             bool add_special, bool parse_special) {
    std::vector<llama_token> toks(content.size() + 16);
    int n = llama_tokenize(S.vocab, content.c_str(), (int32_t) content.size(),
                           toks.data(), (int32_t) toks.size(), add_special, parse_special);
    if (n < 0) {
        toks.resize(-n);
        n = llama_tokenize(S.vocab, content.c_str(), (int32_t) content.size(),
                           toks.data(), (int32_t) toks.size(), add_special, parse_special);
    }
    toks.resize(std::max(n, 0));
    return toks;
}

// apply the model's chat template (chatml fallback) to OpenAI-style messages
static std::string apply_template_str(const jlens_server & S, const json & messages, bool add_assistant) {
    std::vector<std::string> stash;
    for (const auto & m : messages) {
        stash.push_back(m.at("role").get<std::string>());
        stash.push_back(m.at("content").get<std::string>());
    }
    std::vector<llama_chat_message> msgs;
    for (size_t i = 0; i < stash.size(); i += 2) {
        msgs.push_back({stash[i].c_str(), stash[i + 1].c_str()});
    }
    const char * tmpl = llama_model_chat_template(S.model, nullptr);
    std::string buf(4096 + dump_json(messages).size() * 2, '\0');
    int n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_assistant,
                                      buf.data(), (int32_t) buf.size());
    if (n < 0) {
        n = llama_chat_apply_template("chatml", msgs.data(), msgs.size(), add_assistant,
                                      buf.data(), (int32_t) buf.size());
        if (n < 0) throw std::runtime_error("chat template application failed");
    }
    if ((size_t) n > buf.size()) {
        buf.resize(n);
        n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_assistant,
                                      buf.data(), (int32_t) buf.size());
    }
    buf.resize(std::max(n, 0));
    return buf;
}

// parse a JSON array of intervention specs (shared by /jlens/forward and the live set)
static std::vector<intervention> parse_interventions(const jlens_server & S, const json & arr) {
    std::vector<intervention> ivs;
    for (const auto & j : arr) {
        intervention iv;
        iv.layer = j.at("layer").get<int>();
        if (iv.layer < 0) iv.layer += S.n_layer;
        if (iv.layer < 0 || iv.layer >= S.n_layer) throw std::runtime_error("intervention layer out of range");
        iv.p0 = j.value("pos_start", 0);
        iv.p1 = j.value("pos_end", -1);
        const std::string mode = j.value("mode", "add");
        std::vector<uint8_t> raw = base64_decode(j.at("data").get<std::string>());
        if (raw.size() % sizeof(float) != 0) throw std::runtime_error("intervention data not f32-aligned");
        const size_t nf = raw.size() / sizeof(float);
        const float * f = (const float *) raw.data();
        const size_t d = (size_t) S.n_embd;
        if (mode == "add" || mode == "set") {
            iv.mode = mode == "add" ? iv_mode::ADD : iv_mode::SET;
            if (nf != d) throw std::runtime_error("add/set data must be d_model floats");
            iv.a.assign(f, f + d);
        } else if (mode == "lowrank") {
            iv.mode = iv_mode::LOWRANK;
            iv.k = j.at("k").get<int>();
            if (iv.k <= 0 || iv.k > 256) throw std::runtime_error("lowrank k out of range");
            if (nf != d * (size_t) iv.k * 2) throw std::runtime_error("lowrank data must be 2*k*d_model floats (A then B)");
            iv.a.assign(f, f + d * iv.k);
            iv.b.assign(f + d * iv.k, f + 2 * d * iv.k);
        } else {
            throw std::runtime_error("unknown intervention mode: " + mode);
        }
        for (const float * v = f; v < f + nf; v++) {
            if (!std::isfinite(*v)) throw std::runtime_error("intervention data contains non-finite values");
        }
        ivs.push_back(std::move(iv));
    }
    return ivs;
}

// run one decode over tokens[i0, i1) at absolute positions [i0, i1).
// all_logits controls whether every position requests logits (needed for
// full-width last-layer capture and for logits_positions extraction).
static int decode_chunk(jlens_server & S, const std::vector<llama_token> & tokens,
                        int64_t i0, int64_t i1, bool all_logits) {
    const int64_t n = i1 - i0;
    llama_batch batch = llama_batch_init((int32_t) n, 0, 1);
    for (int64_t i = 0; i < n; i++) {
        batch.token[i]    = tokens[i0 + i];
        batch.pos[i]      = (llama_pos) (i0 + i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = all_logits || (i == n - 1);
    }
    batch.n_tokens = (int32_t) n;
    S.cb.base_pos = i0;
    S.cb.expect_n = n;
    const int rc = llama_decode(S.ctx, batch);
    llama_batch_free(batch);
    return rc;
}

// ---------------------------------------------------------------------------
// /jlens/forward
// ---------------------------------------------------------------------------

static void handle_forward(jlens_server & S, const httplib::Request & req, httplib::Response & res) {
    const auto t_start = ggml_time_us();

    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception & e) {
        res.status = 400;
        res.set_content(dump_json(json{{"error", std::string("bad json: ") + e.what()}}), "application/json");
        return;
    }

    try {
        if (!body.contains("tokens") || !body["tokens"].is_array()) {
            throw std::runtime_error("'tokens' (array of ints) is required");
        }
        std::vector<llama_token> tokens = body["tokens"].get<std::vector<llama_token>>();
        if (tokens.empty()) throw std::runtime_error("'tokens' must be non-empty");
        for (llama_token t : tokens) {
            if (t < 0 || t >= S.n_vocab) throw std::runtime_error("token id out of range: " + std::to_string(t));
        }

        const int n_predict = body.value("n_predict", 0);
        if ((int64_t) tokens.size() + n_predict > S.n_ctx) {
            throw std::runtime_error("tokens + n_predict exceeds context size " + std::to_string(S.n_ctx));
        }

        const std::string dtype = body.value("dtype", "f16");
        if (dtype != "f16" && dtype != "f32") throw std::runtime_error("dtype must be f16 or f32");

        // capture set
        std::set<int> capture;
        if (body.contains("capture_layers") && !body["capture_layers"].is_null()) {
            for (int l : body["capture_layers"].get<std::vector<int>>()) {
                if (l < 0) l += S.n_layer;
                if (l < 0 || l >= S.n_layer) throw std::runtime_error("capture layer out of range");
                capture.insert(l);
            }
        } else if (body.value("capture", true)) {
            for (int l = 0; l < S.n_layer; l++) capture.insert(l);
        }

        // interventions
        std::vector<intervention> ivs;
        if (body.contains("interventions")) {
            ivs = parse_interventions(S, body["interventions"]);
        }

        // logits rows requested. Negative indices are resolved against the
        // ACTUAL generated length after decoding (not n_prompt + n_predict),
        // so e.g. -1 is the true final position even when generation stops
        // early on EOG.
        std::set<int64_t>    want_logits;      // non-negative absolute positions
        std::vector<int64_t> want_logits_neg;  // negative offsets, resolved later
        if (body.contains("logits_positions")) {
            for (int64_t p : body["logits_positions"].get<std::vector<int64_t>>()) {
                if (p < 0) want_logits_neg.push_back(p);
                else       want_logits.insert(p);
            }
        }

        // sampling
        json sampling = body.value("sampling", json::object());
        const bool  greedy = sampling.value("greedy", true);
        const float temp   = sampling.value("temp", 0.8f);
        const int   top_k  = sampling.value("top_k", 40);
        const float top_p  = sampling.value("top_p", 0.95f);
        const uint32_t seed = (uint32_t) sampling.value("seed", -1);

        // last-layer capture (and any logits request) needs all-position logits
        const bool all_logits = capture.count(S.n_layer - 1) > 0 ||
                                !want_logits.empty() || !want_logits_neg.empty();

        // ------------- run (serialized) -------------
        std::lock_guard<std::mutex> lock(S.mutex);

        S.cb.reset_request();
        S.cb.capture_layers = capture;
        S.cb.interventions  = std::move(ivs);

        llama_memory_clear(llama_get_memory(S.ctx), true);
        S.cache_tokens.clear();   // the /v1 completion prefix cache is gone now

        std::map<int64_t, std::vector<float>> logits_rows;

        const int64_t n_prompt = (int64_t) tokens.size();
        const auto t_prompt0 = ggml_time_us();
        for (int64_t i0 = 0; i0 < n_prompt; i0 += S.chunk) {
            const int64_t i1 = std::min<int64_t>(i0 + S.chunk, n_prompt);
            const int rc = decode_chunk(S, tokens, i0, i1, all_logits);
            if (rc != 0) throw std::runtime_error("llama_decode failed with code " + std::to_string(rc));
            if (!S.cb.error.empty()) throw std::runtime_error("capture error: " + S.cb.error);
            if (all_logits) {
                for (int64_t p : want_logits) {
                    if (p >= i0 && p < i1) {
                        const float * row = llama_get_logits_ith(S.ctx, (int32_t) (p - i0));
                        logits_rows[p].assign(row, row + S.n_vocab);
                    }
                }
            }
        }
        const auto t_prompt1 = ggml_time_us();

        // ------------- generation -------------
        json generated = json::array();
        llama_sampler * smpl = nullptr;
        if (n_predict > 0) {
            auto sparams = llama_sampler_chain_default_params();
            smpl = llama_sampler_chain_init(sparams);
            if (greedy || temp <= 0.0f) {
                llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
            } else {
                llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
                llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
                llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
                llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
            }
        }
        int n_gen = 0;
        if (n_predict > 0) {
            // index of the last prompt token's logits row in the last chunk
            int32_t last_idx = all_logits ? (int32_t) ((n_prompt - 1) % S.chunk) : -1;
            // when !all_logits, only the final token of each chunk had logits;
            // llama_get_logits_ith(-1)-style: use output index of last logit row
            for (int step = 0; step < n_predict; step++) {
                const int32_t idx = (step == 0) ? (all_logits ? last_idx : (int32_t) -1) : 0;
                llama_token tok = llama_sampler_sample(smpl, S.ctx, idx);
                if (llama_vocab_is_eog(S.vocab, tok)) break;
                tokens.push_back(tok);
                generated.push_back(json{{"token", tok}, {"piece", piece_json(S.token_piece(tok, true))}});
                n_gen++;
                const int64_t pos = n_prompt + step;
                const int rc = decode_chunk(S, tokens, pos, pos + 1, true);
                if (rc != 0) throw std::runtime_error("llama_decode (gen) failed with code " + std::to_string(rc));
                if (!S.cb.error.empty()) throw std::runtime_error("capture error: " + S.cb.error);
                if (want_logits.count(pos)) {
                    // same semantics as prompt positions: logits at p are the
                    // next-token distribution after consuming token p
                    const float * row = llama_get_logits_ith(S.ctx, 0);
                    logits_rows[pos].assign(row, row + S.n_vocab);
                }
            }
        }
        if (smpl) llama_sampler_free(smpl);
        const auto t_gen1 = ggml_time_us();

        // resolve negative logits_positions against the actual length. The
        // logits of the final decoded position are still live: index 0 when a
        // token was just generated, else the last prompt token's row.
        if (!want_logits_neg.empty()) {
            const int64_t total = n_prompt + n_gen;
            const int32_t final_idx =
                (n_gen > 0) ? 0 : (all_logits ? (int32_t) ((n_prompt - 1) % S.chunk) : -1);
            for (int64_t o : want_logits_neg) {
                const int64_t pos = total + o;
                if (pos < 0 || logits_rows.count(pos)) continue;
                if (pos == total - 1) {  // the common case (-1); earlier ones aren't retained
                    const float * row = llama_get_logits_ith(S.ctx, final_idx);
                    logits_rows[pos].assign(row, row + S.n_vocab);
                }
            }
        }

        // ------------- assemble binary response -------------
        const int64_t n_total = n_prompt + n_gen;
        const size_t  d = (size_t) S.n_embd;

        std::string payload;
        json acts_meta = json::array();
        for (auto & kv : S.cb.captured) {
            const int layer = kv.first;
            std::vector<float> & data = kv.second;
            const int64_t n_pos = (int64_t) (data.size() / d);
            if (n_pos != n_total) {
                throw std::runtime_error("layer " + std::to_string(layer) + " captured " +
                                         std::to_string(n_pos) + " positions, expected " + std::to_string(n_total));
            }
            const size_t offset = payload.size();
            size_t nbytes;
            if (dtype == "f16") {
                std::vector<ggml_fp16_t> half(data.size());
                ggml_fp32_to_fp16_row(data.data(), half.data(), (int64_t) data.size());
                nbytes = half.size() * sizeof(ggml_fp16_t);
                payload.append((const char *) half.data(), nbytes);
            } else {
                nbytes = data.size() * sizeof(float);
                payload.append((const char *) data.data(), nbytes);
            }
            acts_meta.push_back(json{
                {"layer", layer}, {"dtype", dtype},
                {"shape", json::array({n_pos, (int64_t) d})},
                {"offset", offset}, {"nbytes", nbytes},
            });
        }
        json logits_meta = json::array();
        for (auto & kv : logits_rows) {
            const size_t offset = payload.size();
            const size_t nbytes = kv.second.size() * sizeof(float);
            payload.append((const char *) kv.second.data(), nbytes);
            logits_meta.push_back(json{{"pos", kv.first}, {"offset", offset}, {"nbytes", nbytes}});
        }

        json header = {
            {"tokens", tokens},
            {"n_prompt", n_prompt},
            {"n_gen", n_gen},
            {"generated", generated},
            {"activations", acts_meta},
            {"logits", logits_meta},
            {"timings", {
                {"prompt_ms", (t_prompt1 - t_prompt0) / 1000.0},
                {"gen_ms", (t_gen1 - t_prompt1) / 1000.0},
                {"total_ms", (ggml_time_us() - t_start) / 1000.0},
            }},
        };
        const std::string hdr = dump_json(header);

        std::string out;
        out.reserve(12 + hdr.size() + payload.size());
        out.append("JLNS", 4);
        const uint32_t version = 1;
        const uint32_t hdr_len = (uint32_t) hdr.size();
        out.append((const char *) &version, 4);
        out.append((const char *) &hdr_len, 4);
        out.append(hdr);
        out.append(payload);
        res.set_content(std::move(out), "application/octet-stream");
    } catch (const std::exception & e) {
        res.status = 400;
        res.set_content(dump_json(json{{"error", e.what()}}), "application/json");
    }
}

// ---------------------------------------------------------------------------
// backend mode: OpenAI-compatible completions with the live intervention set
// ---------------------------------------------------------------------------

struct gen_params {
    int      max_tokens = 512;
    float    temp       = 0.8f;
    float    top_p      = 0.95f;
    int      top_k      = 40;
    uint32_t seed       = (uint32_t) -1;
    std::vector<std::string> stop;

    static gen_params from_request(const json & body) {
        gen_params gp;
        gp.max_tokens = body.value("max_tokens", body.value("max_completion_tokens", 512));
        gp.temp  = body.value("temperature", 0.8f);
        gp.top_p = body.value("top_p", 0.95f);
        gp.top_k = body.value("top_k", 40);
        gp.seed  = (uint32_t) body.value("seed", -1);
        if (body.contains("stop")) {
            if (body["stop"].is_string()) gp.stop.push_back(body["stop"].get<std::string>());
            else if (body["stop"].is_array()) {
                for (const auto & s : body["stop"]) gp.stop.push_back(s.get<std::string>());
            }
        }
        return gp;
    }
};

// Generate a completion with the live intervention set active. Caller must
// hold S.mutex. Reuses the KV prefix from the previous completion when the
// prompt extends it and the intervention set is unchanged (multi-turn chat).
// `emit` (optional) receives stop-scanned text increments for streaming.
static json run_completion(jlens_server & S, const std::vector<llama_token> & prompt,
                           const gen_params & gp,
                           const std::function<void(const std::string &)> & emit) {
    const auto t0 = ggml_time_us();
    if (prompt.empty()) throw std::runtime_error("empty prompt");
    if ((int64_t) prompt.size() >= S.n_ctx) {
        throw std::runtime_error("prompt (" + std::to_string(prompt.size()) +
                                 " tokens) does not fit the context (" + std::to_string(S.n_ctx) + ")");
    }
    const int max_tokens = std::min<int64_t>(gp.max_tokens, S.n_ctx - (int64_t) prompt.size());

    // ---- KV prefix reuse ----
    const size_t iv_hash = std::hash<std::string>{}(S.live_iv_raw);
    if (iv_hash != S.cache_iv_hash) {
        llama_memory_clear(llama_get_memory(S.ctx), true);
        S.cache_tokens.clear();
        S.cache_iv_hash = iv_hash;
    }
    size_t common = 0;
    while (common < S.cache_tokens.size() && common < prompt.size() &&
           S.cache_tokens[common] == prompt[common]) common++;
    if (common == prompt.size()) common = prompt.size() - 1;  // must decode >=1 token for logits
    // Always trim seq 0 to the reused prefix. Unconditional (not just when
    // common < cache_tokens.size()) because a prior /jlens/forward may have
    // left KV in seq 0 without updating cache_tokens.
    llama_memory_seq_rm(llama_get_memory(S.ctx), 0, (llama_pos) common, -1);
    S.cache_tokens.resize(common);

    // interventions on, capture off
    S.cb.reset_request();
    S.cb.interventions = S.live_ivs;

    std::vector<llama_token> all_tokens = prompt;
    for (size_t i0 = common; i0 < prompt.size(); i0 += S.chunk) {
        const size_t i1 = std::min(i0 + (size_t) S.chunk, prompt.size());
        const int rc = decode_chunk(S, all_tokens, (int64_t) i0, (int64_t) i1, false);
        if (rc != 0) throw std::runtime_error("llama_decode failed with code " + std::to_string(rc));
        if (!S.cb.error.empty()) throw std::runtime_error(S.cb.error);
    }
    const auto t_prompt = ggml_time_us();

    // ---- sampling ----
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (gp.temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(gp.top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(gp.top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(gp.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(gp.seed));
    }

    stop_scanner scanner(gp.stop);
    std::string content;
    std::string finish = "length";
    int n_gen = 0;
    size_t n_decoded = prompt.size();

    for (int step = 0; step < max_tokens; step++) {
        const llama_token tok = llama_sampler_sample(smpl, S.ctx, -1);
        if (llama_vocab_is_eog(S.vocab, tok)) { finish = "stop"; break; }
        all_tokens.push_back(tok);
        n_gen++;
        const std::string out = scanner.feed(S.token_piece(tok, false));
        if (!out.empty()) {
            content += out;
            if (emit) emit(out);
        }
        if (scanner.hit) { finish = "stop"; break; }
        const int64_t pos = (int64_t) prompt.size() + step;
        const int rc = decode_chunk(S, all_tokens, pos, pos + 1, false);
        if (rc != 0) { llama_sampler_free(smpl); throw std::runtime_error("llama_decode (gen) failed with code " + std::to_string(rc)); }
        if (!S.cb.error.empty()) { llama_sampler_free(smpl); throw std::runtime_error(S.cb.error); }
        n_decoded++;
    }
    llama_sampler_free(smpl);
    const std::string tail = scanner.flush();
    if (!tail.empty()) {
        content += tail;
        if (emit) emit(tail);
    }

    // cache exactly the tokens whose KV is in memory
    S.cache_tokens.assign(all_tokens.begin(), all_tokens.begin() + n_decoded);

    S.completion_counter++;
    S.last_completion = {
        {"id", S.completion_counter},
        {"tokens", all_tokens},
        {"n_prompt", prompt.size()},
        {"n_gen", n_gen},
        {"text", piece_json(content)},
        {"finish_reason", finish},
        {"n_cached", common},
        {"interventions_active", S.live_ivs.size()},
    };

    return {
        {"content", content},
        {"finish_reason", finish},
        {"n_prompt", prompt.size()},
        {"n_gen", n_gen},
        {"timings", {
            {"cached_tokens", common},
            {"prompt_ms", (t_prompt - t0) / 1000.0},
            {"total_ms", (ggml_time_us() - t0) / 1000.0},
        }},
    };
}

static json oai_error(const std::string & msg) {
    return {{"error", {{"message", msg}, {"type", "invalid_request_error"}}}};
}

// register /v1/* + live intervention endpoints
static void register_backend_endpoints(httplib::Server & svr, jlens_server & S) {
    svr.Get("/v1/models", [&S](const httplib::Request &, httplib::Response & res) {
        res.set_content(dump_json({
            {"object", "list"},
            {"data", {{{"id", S.model_name}, {"object", "model"}, {"owned_by", "jlens-gguf"}}}},
        }), "application/json");
    });

    svr.Get("/jlens/interventions", [&S](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(S.mutex);
        // echo the specs without the (bulky) vector payloads
        json slim = json::array();
        for (auto j : json::parse(S.live_iv_raw)) {
            j.erase("data");
            slim.push_back(std::move(j));
        }
        res.set_content(dump_json({
            {"count", S.live_ivs.size()},
            {"interventions", slim},
            {"meta", S.live_iv_meta},
        }), "application/json");
    });

    auto set_live = [&S](const httplib::Request & req, httplib::Response & res) {
        try {
            json body = req.body.empty() ? json{{"interventions", json::array()}} : json::parse(req.body);
            json arr = body.value("interventions", json::array());
            auto parsed = parse_interventions(S, arr);
            std::lock_guard<std::mutex> lock(S.mutex);
            S.live_ivs = std::move(parsed);
            // keep the FULL spec (incl. vector data) — its hash drives KV-cache
            // invalidation, and an alpha-only change must invalidate too
            S.live_iv_raw = dump_json(arr);
            S.live_iv_meta = body.value("meta", json());
            res.set_content(dump_json({{"count", S.live_ivs.size()}}), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(dump_json({{"error", e.what()}}), "application/json");
        }
    };
    svr.Post("/jlens/interventions", set_live);
    svr.Delete("/jlens/interventions", [&S](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(S.mutex);
        S.live_ivs.clear();
        S.live_iv_raw = "[]";
        S.live_iv_meta = json();
        res.set_content(dump_json({{"count", 0}}), "application/json");
    });

    svr.Get("/jlens/last_completion", [&S](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(S.mutex);
        res.set_content(dump_json(S.last_completion.is_null()
                                  ? json{{"id", 0}} : S.last_completion), "application/json");
    });

    // shared handler body for both /v1 endpoints; `chat` picks the schema
    auto completions = [&S](const httplib::Request & req, httplib::Response & res, bool chat) {
        json body;
        std::vector<llama_token> prompt;
        try {
            body = json::parse(req.body);
            if (chat) {
                if (!body.contains("messages")) throw std::runtime_error("'messages' is required");
                prompt = tokenize_str(S, apply_template_str(S, body["messages"], true), true, true);
            } else {
                if (!body.contains("prompt") || !body["prompt"].is_string()) {
                    throw std::runtime_error("'prompt' (string) is required");
                }
                prompt = tokenize_str(S, body["prompt"].get<std::string>(), true, true);
            }
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(dump_json(oai_error(e.what())), "application/json");
            return;
        }
        const gen_params gp = gen_params::from_request(body);
        const bool stream = body.value("stream", false);
        const std::string cmpl_id = (chat ? "chatcmpl-" : "cmpl-") + std::to_string(ggml_time_us());
        const int64_t created = (int64_t) time(nullptr);
        const std::string object = chat ? "chat.completion" : "text_completion";

        auto full_response = [&, chat](const json & r) -> json {
            json choice = chat
                ? json{{"index", 0},
                       {"message", {{"role", "assistant"}, {"content", piece_json(r["content"].get<std::string>())}}},
                       {"finish_reason", r["finish_reason"]}}
                : json{{"index", 0},
                       {"text", piece_json(r["content"].get<std::string>())},
                       {"finish_reason", r["finish_reason"]}};
            return {
                {"id", cmpl_id}, {"object", object}, {"created", created}, {"model", S.model_name},
                {"choices", json::array({choice})},
                {"usage", {{"prompt_tokens", r["n_prompt"]}, {"completion_tokens", r["n_gen"]},
                           {"total_tokens", (int64_t) r["n_prompt"] + (int64_t) r["n_gen"]}}},
                {"timings", r["timings"]},
            };
        };

        if (!stream) {
            try {
                std::lock_guard<std::mutex> lock(S.mutex);
                json r = run_completion(S, prompt, gp, nullptr);
                res.set_content(dump_json(full_response(r)), "application/json");
            } catch (const std::exception & e) {
                res.status = 500;
                res.set_content(dump_json(oai_error(e.what())), "application/json");
            }
            return;
        }

        // streaming: the provider runs after this handler returns — capture by value
        const std::string chunk_object = chat ? "chat.completion.chunk" : "text_completion";
        auto make_chunk = [cmpl_id, created, chunk_object, chat, name = S.model_name]
                          (const json & delta_or_text, const json & finish) -> std::string {
            json choice = chat
                ? json{{"index", 0}, {"delta", delta_or_text}, {"finish_reason", finish}}
                : json{{"index", 0}, {"text", delta_or_text}, {"finish_reason", finish}};
            return "data: " + dump_json({
                {"id", cmpl_id}, {"object", chunk_object}, {"created", created}, {"model", name},
                {"choices", json::array({choice})},
            }) + "\n\n";
        };
        res.set_header("Cache-Control", "no-store");
        res.set_chunked_content_provider("text/event-stream",
            [&S, prompt, gp, chat, make_chunk](size_t, httplib::DataSink & sink) {
                auto send = [&](const std::string & s) { sink.write(s.data(), s.size()); };
                try {
                    if (chat) send(make_chunk({{"role", "assistant"}}, nullptr));
                    std::lock_guard<std::mutex> lock(S.mutex);
                    json r = run_completion(S, prompt, gp, [&](const std::string & piece) {
                        send(make_chunk(chat ? json{{"content", piece_json(piece)}} : piece_json(piece), nullptr));
                    });
                    send(make_chunk(chat ? json::object() : json(""), r["finish_reason"]));
                } catch (const std::exception & e) {
                    send("data: " + dump_json(oai_error(e.what())) + "\n\n");
                }
                send("data: [DONE]\n\n");
                sink.done();
                return true;
            });
    };
    svr.Post("/v1/chat/completions", [completions](const httplib::Request & req, httplib::Response & res) {
        completions(req, res, true);
    });
    svr.Post("/v1/completions", [completions](const httplib::Request & req, httplib::Response & res) {
        completions(req, res, false);
    });
    svr.Post("/completion", [completions](const httplib::Request & req, httplib::Response & res) {
        // minimal llama-server-style alias (non-streaming shape differs; OpenAI schema returned)
        completions(req, res, false);
    });
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void print_usage(const char * prog) {
    fprintf(stderr,
        "usage: %s -m MODEL.gguf [options]\n"
        "\n"
        "jlens-server is a drop-in, steerable llama-server: it accepts the common\n"
        "llama-server launch flags, so you can usually swap the binary name and go.\n"
        "\n"
        "  -m, --model PATH            GGUF model (required)\n"
        "      --host HOST             bind address (default 127.0.0.1)\n"
        "      --port PORT             port (default 8091)\n"
        "  -c, --ctx-size N            context size (default 4096)\n"
        "  -b, --batch-size N          logical batch = prompt chunk (default 512)\n"
        "  -ub,--ubatch-size N         physical batch (alias of --chunk)\n"
        "      --chunk N               prompt-processing chunk (llama-server -b/-ub)\n"
        "  -t, --threads N             threads (default: hardware)\n"
        "  -ngl,--n-gpu-layers N       layers to offload (default 0; CPU capture is exact)\n"
        "      --gpu-layers N          alias of --n-gpu-layers\n"
        "  -mg,--main-gpu N            main GPU index\n"
        "  -fa,--flash-attn [on|off|auto]  Flash Attention (default auto)\n"
        "      --no-mmap               do not memory-map the model\n"
        "      --mlock                 lock the model in RAM\n"
        "  -h, --help\n"
        "\n"
        "Accepted for llama-server compatibility but ignored (jlens-server is a\n"
        "single-sequence introspection server): --parallel/-np, --cont-batching,\n"
        "--api-key, --jinja, --embeddings, --alias, --verbose, --log-* , -ngld, etc.\n",
        prog);
}

// flags that take one value and are safely ignored (llama-server superset)
static bool is_ignored_valued_flag(const std::string & a) {
    static const std::set<std::string> flags = {
        "-np", "--parallel", "--api-key", "--api-key-file", "-a", "--alias",
        "--rope-scaling", "--rope-freq-base", "--rope-freq-scale", "--yarn-orig-ctx",
        "--yarn-ext-factor", "--yarn-attn-factor", "--yarn-beta-slow", "--yarn-beta-fast",
        "--cache-type-k", "-ctk", "--cache-type-v", "-ctv", "--n-predict", "-n",
        "--keep", "--split-mode", "-sm", "--tensor-split", "-ts", "--numa",
        "--override-kv", "--lora", "--lora-scaled", "--seed", "-s", "--log-file",
        "--timeout", "--threads-batch", "-tb", "--poll", "--grp-attn-n", "--grp-attn-w",
        "--defrag-thold", "-dt", "--slots", "--metrics", "--slot-save-path", "--draft",
        "--model-draft", "-md", "--gpu-layers-draft", "-ngld", "--device", "-dev",
    };
    return flags.count(a) > 0;
}
static bool is_ignored_bool_flag(const std::string & a) {
    static const std::set<std::string> flags = {
        "--cont-batching", "-cb", "--no-cont-batching", "--jinja", "--embeddings",
        "--embedding", "--verbose", "-v", "--verbose-prompt", "--log-disable",
        "--log-colors", "--no-webui", "--metrics", "--flash-attn-off", "--mlock",
        "--no-kv-offload", "-nkvo", "--cont", "--special", "--no-perf",
    };
    return flags.count(a) > 0;
}

int main(int argc, char ** argv) {
    std::string model_path, host = "127.0.0.1";
    int port = 8091, n_ctx = 4096, chunk = 512;
    int n_threads = (int) std::thread::hardware_concurrency();
    int n_gpu_layers = 0, main_gpu = 0;
    bool use_mmap = true, use_mlock = false;
    llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char * flag) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", flag); exit(1); }
            return argv[++i];
        };
        if      (a == "-m" || a == "--model")     model_path = next(a.c_str());
        else if (a == "--host")                   host = next(a.c_str());
        else if (a == "--port")                   port = atoi(next(a.c_str()).c_str());
        else if (a == "-c" || a == "--ctx-size")  n_ctx = atoi(next(a.c_str()).c_str());
        else if (a == "--chunk" || a == "-b" || a == "--batch-size" ||
                 a == "-ub" || a == "--ubatch-size")
                                                  chunk = atoi(next(a.c_str()).c_str());
        else if (a == "-t" || a == "--threads")   n_threads = atoi(next(a.c_str()).c_str());
        else if (a == "-ngl" || a == "--n-gpu-layers" || a == "--gpu-layers")
                                                  n_gpu_layers = atoi(next(a.c_str()).c_str());
        else if (a == "-mg" || a == "--main-gpu") main_gpu = atoi(next(a.c_str()).c_str());
        else if (a == "-fa" || a == "--flash-attn") {
            // llama-server accepts -fa with or without a value (on/off/auto)
            std::string v = (i + 1 < argc && argv[i + 1][0] != '-') ? argv[++i] : "on";
            flash_attn = (v == "off" || v == "0" || v == "disabled") ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                       : (v == "auto") ? LLAMA_FLASH_ATTN_TYPE_AUTO
                       : LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        else if (a == "--no-mmap")                use_mmap = false;
        else if (a == "--mlock")                  use_mlock = true;
        else if (a == "-h" || a == "--help")      { print_usage(argv[0]); return 0; }
        else if (is_ignored_valued_flag(a))       { std::string v = next(a.c_str()); fprintf(stderr, "jlens-server: ignoring %s %s\n", a.c_str(), v.c_str()); }
        else if (is_ignored_bool_flag(a))         { fprintf(stderr, "jlens-server: ignoring %s\n", a.c_str()); }
        else { fprintf(stderr, "unknown arg: %s (see --help; llama-server superset)\n", a.c_str()); print_usage(argv[0]); return 1; }
    }
    if (model_path.empty()) { print_usage(argv[0]); return 1; }

    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
    }, nullptr);

    jlens_server S;
    S.model_path = model_path;
    S.n_ctx = n_ctx;
    S.chunk = chunk;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.main_gpu     = main_gpu;
    mparams.use_mmap     = use_mmap;
    mparams.use_mlock    = use_mlock;
    S.model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!S.model) { fprintf(stderr, "error: failed to load model %s\n", model_path.c_str()); return 1; }
    S.vocab = llama_model_get_vocab(S.model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_ctx;
    cparams.n_batch         = chunk;
    cparams.n_ubatch        = chunk;
    cparams.flash_attn_type = flash_attn;
    cparams.cb_eval         = jlens_cb_eval;
    cparams.cb_eval_user_data = &S.cb;
    S.ctx = llama_init_from_model(S.model, cparams);
    if (!S.ctx) { fprintf(stderr, "error: failed to create context\n"); return 1; }
    llama_set_n_threads(S.ctx, n_threads, n_threads);

    S.n_layer = llama_model_n_layer(S.model);
    S.n_embd  = llama_model_n_embd(S.model);
    S.n_vocab = llama_vocab_n_tokens(S.vocab);

    // startup self-check: verify l_out capture works on this architecture
    {
        S.cb.reset_request();
        for (int l = 0; l < S.n_layer; l++) S.cb.capture_layers.insert(l);
        std::vector<llama_token> probe = { llama_vocab_bos(S.vocab) >= 0 ? llama_vocab_bos(S.vocab) : 0 };
        llama_memory_clear(llama_get_memory(S.ctx), true);
        if (decode_chunk(S, probe, 0, 1, true) == 0) {
            S.l_out_ok = (int) S.cb.captured.size() == S.n_layer && S.cb.error.empty();
        }
        llama_memory_clear(llama_get_memory(S.ctx), true);
        S.cb.reset_request();
        if (!S.l_out_ok) {
            fprintf(stderr, "error: captured %zu/%d layers in self-check (%s); "
                    "this architecture may not expose l_out tensors\n",
                    S.cb.captured.size(), S.n_layer, S.cb.error.c_str());
        }
    }

    // cache /vocab
    {
        json pieces = json::array();
        json attrs  = json::array();
        for (int t = 0; t < S.n_vocab; t++) {
            pieces.push_back(piece_json(S.token_piece(t, true)));
            attrs.push_back((int) llama_vocab_get_attr(S.vocab, t));
        }
        S.vocab_json = dump_json(json{{"n_vocab", S.n_vocab}, {"pieces", pieces}, {"attrs", attrs}});
    }

    char desc[256] = {0};
    llama_model_desc(S.model, desc, sizeof(desc));
    {
        char name[256] = {0};
        if (llama_model_meta_val_str(S.model, "general.name", name, sizeof(name)) > 0) {
            S.model_name = name;
        } else {
            const size_t slash = model_path.find_last_of('/');
            S.model_name = slash == std::string::npos ? model_path : model_path.substr(slash + 1);
        }
    }

    httplib::Server svr;
    svr.set_default_headers({{"Access-Control-Allow-Origin", "*"},
                             {"Access-Control-Allow-Headers", "Content-Type"}});
    svr.Options(".*", [](const httplib::Request &, httplib::Response & res) { res.status = 204; });

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(dump_json(json{{"status", "ok"}}), "application/json");
    });

    svr.Get("/props", [&](const httplib::Request &, httplib::Response & res) {
        const char * tmpl = llama_model_chat_template(S.model, nullptr);
        res.set_content(dump_json(json{
            {"model_path", S.model_path},
            {"model_desc", desc},
            {"n_layer", S.n_layer},
            {"n_embd", S.n_embd},
            {"n_vocab", S.n_vocab},
            {"n_ctx", S.n_ctx},
            {"chunk", S.chunk},
            {"l_out_ok", S.l_out_ok},
            {"llama_commit", JLENS_LLAMA_COMMIT},
            {"bos", llama_vocab_bos(S.vocab)},
            {"eos", llama_vocab_eos(S.vocab)},
            {"add_bos", llama_vocab_get_add_bos(S.vocab)},
            {"has_chat_template", tmpl != nullptr},
        }), "application/json");
    });

    svr.Get("/vocab", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(S.vocab_json, "application/json");
    });

    svr.Post("/tokenize", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            json body = json::parse(req.body);
            const std::string content = body.at("content").get<std::string>();
            const bool add_special   = body.value("add_special", true);
            const bool parse_special = body.value("parse_special", true);
            std::vector<llama_token> toks(content.size() + 16);
            int n = llama_tokenize(S.vocab, content.c_str(), (int32_t) content.size(),
                                   toks.data(), (int32_t) toks.size(), add_special, parse_special);
            if (n < 0) { toks.resize(-n); n = llama_tokenize(S.vocab, content.c_str(), (int32_t) content.size(),
                                   toks.data(), (int32_t) toks.size(), add_special, parse_special); }
            toks.resize(std::max(n, 0));
            json pieces = json::array();
            for (llama_token t : toks) pieces.push_back(piece_json(S.token_piece(t, true)));
            res.set_content(dump_json(json{{"tokens", toks}, {"pieces", pieces}}), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(dump_json(json{{"error", e.what()}}), "application/json");
        }
    });

    svr.Post("/detokenize", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            json body = json::parse(req.body);
            std::vector<llama_token> toks = body.at("tokens").get<std::vector<llama_token>>();
            std::string text(std::max<size_t>(toks.size() * 8, 256), '\0');
            int n = llama_detokenize(S.vocab, toks.data(), (int32_t) toks.size(),
                                     text.data(), (int32_t) text.size(), false, true);
            if (n < 0) { text.resize(-n); n = llama_detokenize(S.vocab, toks.data(), (int32_t) toks.size(),
                                     text.data(), (int32_t) text.size(), false, true); }
            text.resize(std::max(n, 0));
            res.set_content(dump_json(json{{"content", piece_json(text)}}), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(dump_json(json{{"error", e.what()}}), "application/json");
        }
    });

    svr.Post("/apply_template", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            json body = json::parse(req.body);
            const std::string prompt = apply_template_str(S, body.at("messages"),
                                                          body.value("add_assistant", true));
            res.set_content(dump_json(json{{"prompt", prompt}}), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(dump_json(json{{"error", e.what()}}), "application/json");
        }
    });

    register_backend_endpoints(svr, S);

    svr.Post("/jlens/forward", [&](const httplib::Request & req, httplib::Response & res) {
        handle_forward(S, req, res);
    });

    fprintf(stderr, "jlens-server: %s\n", desc);
    fprintf(stderr, "jlens-server: n_layer=%d n_embd=%d n_vocab=%d n_ctx=%d chunk=%d threads=%d l_out_ok=%s\n",
            S.n_layer, S.n_embd, S.n_vocab, S.n_ctx, S.chunk, n_threads, S.l_out_ok ? "yes" : "NO");
    fprintf(stderr, "jlens-server: listening on http://%s:%d\n", host.c_str(), port);
    if (!svr.listen(host, port)) {
        fprintf(stderr, "error: failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }

    llama_free(S.ctx);
    llama_model_free(S.model);
    return 0;
}
