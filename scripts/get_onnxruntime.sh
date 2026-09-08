#!/usr/bin/env bash
# Fetches an ONNX Runtime release into third_party/ and prints the ORT_ROOT to
# pass to CMake. No package manager involved -- it is one zip.
set -euo pipefail

VERSION="${1:-1.22.0}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party"
mkdir -p "$DEST"

case "$(uname -s)" in
  Linux*)  PKG="onnxruntime-linux-x64-${VERSION}"; EXT="tgz" ;;
  Darwin*) PKG="onnxruntime-osx-universal2-${VERSION}"; EXT="tgz" ;;
  MINGW*|MSYS*|CYGWIN*) PKG="onnxruntime-win-x64-${VERSION}"; EXT="zip" ;;
  *) echo "unsupported platform: $(uname -s)" >&2; exit 1 ;;
esac

URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${PKG}.${EXT}"

if [ ! -d "$DEST/$PKG" ]; then
  echo "fetching $URL"
  curl -sSL -o "$DEST/$PKG.$EXT" "$URL"
  if [ "$EXT" = "zip" ]; then
    unzip -q -o "$DEST/$PKG.$EXT" -d "$DEST"
  else
    tar -xzf "$DEST/$PKG.$EXT" -C "$DEST"
  fi
  rm -f "$DEST/$PKG.$EXT"
fi

echo "$DEST/$PKG"
