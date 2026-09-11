# Patches against vendored dependencies

Changes to `vendor/llama.cpp` live here as well as in the working tree, because
a submodule update discards them silently. Reapply with:

    cd vendor/llama.cpp && git apply ../../patches/<name>.patch

## llamacpp-gemma4-global-head-dim.patch

`convert_hf_to_gguf.py` reads `hparams["global_head_dim"]` and raises KeyError on
any Gemma 4 checkpoint saved by transformers 5.16.1, which expresses the same
value as `per_layer_config`:

    transformers 5.6.2  (upstream)  "global_head_dim": 512
    transformers 5.16.1 (ours)      "per_layer_config": {"05": {"head_dim": 512}, ...}

Patching the model's own `config.json` does NOT work: `load_hparams` goes through
`AutoConfig.from_pretrained(...).to_dict()`, which normalises the old key back
into the new form, so the added key never reaches the converter. The derivation
therefore has to live in the converter.

It is verified rather than assumed -- `per_layer_config`'s keys must be exactly
the `full_attention` layers and must agree on one `head_dim`, otherwise it
raises. A wrong value here does not crash: it silently writes the wrong
key/value length and RoPE dimension count into the GGUF.

Verified on `gemma4-e4b-coder`: key_length 512, key_length_swa 256,
rope.dimension_count 512, rope.dimension_count_swa 256.
