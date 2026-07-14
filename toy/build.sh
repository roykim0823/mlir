#!/usr/bin/env bash
# Build all chapters (or a single one) with one shared, incremental build tree.
#
# Usage:
#   ./build.sh              # configure once (if needed) + build everything
#   ./build.sh ch3          # build only toyc-ch3
#   ./build.sh --fresh      # wipe build/ and rebuild everything
#   ./build.sh ch5 --fresh  # wipe build/ and build only toyc-ch5
#
# Binaries land in build/bin/toyc-ch{1..7}.
set -euo pipefail
cd "$(dirname "$0")"

target="all"
fresh=0
for arg in "$@"; do
  case "$arg" in
    ch[1-7]) target="$arg" ;;
    all)     target="all" ;;
    --fresh) fresh=1 ;;
    *)
      echo "usage: ./build.sh [ch1..ch7|all] [--fresh]" >&2
      exit 1
      ;;
  esac
done

if [[ $fresh -eq 1 ]]; then
  rm -rf build
fi

# Configure only when there is no existing build tree; afterwards Ninja
# re-runs CMake automatically whenever a CMakeLists.txt changes.
if [[ ! -f build/CMakeCache.txt ]]; then
  cmake --preset default
fi

if [[ "$target" == "all" ]]; then
  cmake --build --preset default
else
  cmake --build --preset default --target "toyc-$target"
fi
