#!/usr/bin/env bash
# The whole pipeline, from nothing to a served slide, in one go.
# Usage: scripts/demo.sh [build-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
cd "$ROOT"
mkdir -p data out results

BIN="$BUILD"
[ -x "$BIN/wsi_scan" ] || BIN="$BUILD/Release"   # multi-config generators
EXT=""
[ -x "$BIN/wsi_scan.exe" ] && EXT=".exe"

echo "== 1. generate a slide with known ground truth"
"$BIN/wsi_makeslide$EXT" --out data/slide.ppm --truth data/slide_truth.json

echo
echo "== 2. unit tests"
"$BIN/wsi_test$EXT"

echo
echo "== 3. inner-loop throughput"
"$BIN/wsi_bench$EXT" 5 | tee results/bench.txt

MODEL=""
if [ -f models/tile_quality.onnx ]; then
  MODEL="--model models/tile_quality.onnx --threads 4"
fi

echo
echo "== 4. scan, verified focus map (default)"
"$BIN/wsi_scan$EXT" --slide data/slide.ppm --truth data/slide_truth.json --out out $MODEL
cp out/report.json results/report_verified.json

echo
echo "== 5. scan, every tile swept (the baseline the map is measured against)"
"$BIN/wsi_scan$EXT" --slide data/slide.ppm --truth data/slide_truth.json --out out \
  --no-map --no-preview $MODEL
cp out/report.json results/report_nomap.json

echo
echo "== 6. scan, map trusted without verification (what the classifier has to catch)"
"$BIN/wsi_scan$EXT" --slide data/slide.ppm --truth data/slide_truth.json --out out \
  --trust-map --no-preview $MODEL
cp out/report.json results/report_trustmap.json

echo
echo "reports in results/; serve the slide with:"
echo "  uvicorn python.service:app --port 8000   # then open http://localhost:8000/"
