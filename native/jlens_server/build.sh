#!/usr/bin/env bash
# Build jlens-server against quant-tuner's vendored llama.cpp.
#
#   bash native/jlens_server/build.sh                 # uses vendor/llama.cpp
#   LLAMA_CPP_DIR=/path/to/llama.cpp bash native/jlens_server/build.sh
#
# Adapted from igorbarshteyn/jlens-gguf (Apache-2.0). jlens-server depends
# ONLY on llama.cpp's PUBLIC API (llama.h, ggml*.h, libllama/libggml). The
# HTTP + JSON deps are vendored under vendor/, so the build does not reach
# into llama.cpp's internals — bumping the submodule pin is just a rebuild.
#
# Unlike upstream, this script expects llama.cpp to be built already (the
# same build quant-tuner's other tooling uses — Metal/CUDA/CPU alike) and
# prints the repo's canonical cmake commands if it is not. Pass --auto-build
# to configure+build the libraries here (CPU-only, upstream behavior).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# honor the same env override paths.py uses (LLAMA_CPP_DIR), then LLAMA_DIR
# for upstream compatibility, then the vendored submodule
LLAMA_DIR="${LLAMA_CPP_DIR:-${LLAMA_DIR:-$REPO_ROOT/vendor/llama.cpp}}"
LLAMA_DIR="$(cd "$LLAMA_DIR" && pwd)"
BUILD_DIR="${LLAMA_BUILD_DIR:-$LLAMA_DIR/build}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
AUTO_BUILD=0
[ "${1:-}" = "--auto-build" ] && AUTO_BUILD=1

if [ ! -f "$LLAMA_DIR/include/llama.h" ]; then
    echo "error: $LLAMA_DIR is not a llama.cpp checkout." >&2
    echo "       Run 'git submodule update --init --recursive' or set LLAMA_CPP_DIR." >&2
    exit 1
fi

# locate libllama (built shared or static)
find_lib() {
    for ext in so dylib a; do
        f=$(ls "$BUILD_DIR"/bin/libllama.$ext "$BUILD_DIR"/src/libllama.$ext \
               "$BUILD_DIR"/libllama.$ext 2>/dev/null | head -1 || true)
        [ -n "$f" ] && { echo "$f"; return; }
    done
}

if [ -z "$(find_lib)" ]; then
    if [ "$AUTO_BUILD" = "1" ]; then
        echo "building llama.cpp in $BUILD_DIR (first time; a few minutes)..."
        cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
              -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
        cmake --build "$BUILD_DIR" -j"$JOBS" --target llama
    else
        echo "error: no libllama found under $BUILD_DIR." >&2
        echo "       Build llama.cpp first (the same build the rest of quant-tuner uses):" >&2
        echo "         cmake -S $LLAMA_DIR -B $BUILD_DIR -DGGML_METAL=ON   # Linux+CUDA: -DGGML_CUDA=ON" >&2
        echo "         cmake --build $BUILD_DIR -j" >&2
        echo "       or re-run with --auto-build for a CPU-only library build." >&2
        exit 1
    fi
fi
LIB_DIR="$(dirname "$(find_lib)")"

# ggml libs can land in a different subdir (build/ggml/src) than libllama
GGML_DIR="$LIB_DIR"
if ! ls "$GGML_DIR"/libggml.* >/dev/null 2>&1; then
    for d in "$BUILD_DIR/ggml/src" "$BUILD_DIR/bin" "$BUILD_DIR"; do
        if ls "$d"/libggml.* >/dev/null 2>&1; then GGML_DIR="$d"; break; fi
    done
fi

LLAMA_COMMIT="$(git -C "$LLAMA_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"

CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O2}"

# vendored cpp-httplib ships split (.h decl + .cpp impl); compile the impl once
HTTPLIB_OBJ="$SCRIPT_DIR/httplib.o"
if [ ! -f "$HTTPLIB_OBJ" ] || [ "$SCRIPT_DIR/vendor/cpp-httplib/httplib.cpp" -nt "$HTTPLIB_OBJ" ]; then
    echo "compiling vendored cpp-httplib..."
    "$CXX" $CXXFLAGS -std=c++17 -pthread -I"$SCRIPT_DIR/vendor/cpp-httplib" \
        -c "$SCRIPT_DIR/vendor/cpp-httplib/httplib.cpp" -o "$HTTPLIB_OBJ"
fi

echo "compiling jlens-server (llama.cpp: $LIB_DIR @ $LLAMA_COMMIT)..."
"$CXX" $CXXFLAGS -std=c++17 -pthread \
    "-DJLENS_LLAMA_COMMIT=\"$LLAMA_COMMIT\"" \
    -I"$LLAMA_DIR/include" -I"$LLAMA_DIR/ggml/include" -I"$SCRIPT_DIR/vendor" \
    -o "$SCRIPT_DIR/jlens-server" "$SCRIPT_DIR/jlens_server.cpp" "$HTTPLIB_OBJ" \
    -L"$LIB_DIR" -L"$GGML_DIR" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$LIB_DIR" -Wl,-rpath,"$GGML_DIR"

echo "built: $SCRIPT_DIR/jlens-server"
"$SCRIPT_DIR/jlens-server" --help 2>&1 | head -2 || true
