# wsi-scan-pipeline

A whole-slide scanner, end to end: **plan the scan → autofocus → acquire →
flat-field → stitch → pyramidal tiled TIFF → classify the tiles → rescan what
came out bad.** C++20 for the pipeline, PyTorch for the tile-quality model,
ONNX Runtime to put that model back inside the C++ scan loop.

The slide is synthetic, and that is the point. Every number below is an **error
against a known truth** — where the tissue is, where the stage really landed,
where the focal plane really was, which tiles really carry a fold. A real `.svs`
has none of that written down, so on real data you can show a picture and say it
looks fine. Here you can be wrong and know it.

```
                    ┌──────────────── the scanner (C++) ──────────────────┐
 slide (ground      │                                                     │
 truth, generated)  │  Otsu ──► serpentine ──► autofocus ──► acquire      │
        │           │  tissue     path over     Brenner /     flat-field  │
        └──────────►│  mask       the tissue    focus map     correction  │
                    │                               ▲            │        │
   ┌── optics ──┐   │                               │            ▼        │
   │ defocus    │   │                          focus map    phase-corr    │
   │ vignette   │──►│                          (plane +     stitching     │
   │ noise      │   │                           local)          │        │
   │ xy jitter  │   │                                            ▼        │
   └────────────┘   │   rescan ◄── tile quality ◄── pyramidal tiled TIFF  │
                    │      ▲        (ONNX Runtime)         │              │
                    └──────┼──────────────────────────────┼──────────────┘
                           │                              ▼
                    tile_quality.onnx            FastAPI /tiles/{l}/{x}/{y}
                    (PyTorch, 61k params)        + Prometheus /metrics
```

Measured on an i5-9300H / GTX 1050 / Windows 11, MSVC 14.44, on a generated
6144×4096 slide (43 FOVs of 512 px acquired). Everything below is reproducible
with `scripts/demo.sh`; the raw reports are in [results/](results/).

---

## The results

### 1. A focus map removes 37% of the autofocus frames and costs nothing

Autofocus is the scan's time budget. A full sweep is ~16 exposures per FOV. A
plane fit through the tiles already focused predicts where the next one will be,
so most FOVs need three frames to *check* the prediction instead of sixteen to
*find* it.

| | full sweep every tile | verified focus map | map trusted blindly |
|---|---:|---:|---:|
| flag | `--no-map` | *(default)* | `--trust-map` |
| AF frames per FOV | 16.23 | **10.14** | 3.49 |
| FOVs needing a full sweep | 100% | **44.2%** | 16.3% |
| AF search, ms per FOV | 185.9 | **103.5** | 36.3 |
| focus error vs truth, mean | 0.30 µm | **0.30 µm** | 1.40 µm |
| focus error vs truth, p95 | 0.64 µm | **0.60 µm** | 3.04 µm |
| simulated instrument time | 17.3 s | **13.3 s** | 13.3 s |

The middle column is the interesting one: **same focus accuracy, 37% fewer
frames.** The right-hand column is what happens if you skip the check — against
the full-sweep baseline, 4.7× fewer frames and a mean error 4.7× worse, because
the map is a smooth surface and the slide is not: the simulated optics carry a
debris bump under the coverslip that no plane fit can see.

The check that makes this safe is not a threshold. Three frames at
`z_pred ± span` are fitted with a parabola; the prediction is accepted only if
the parabola is concave *and its vertex falls inside the span* — the shape of the
triple, not its height. An absolute score threshold fires on every sparse tile at
the rim of a section, because how high a focus metric gets depends on how much
tissue is in the frame.

The span trades accuracy for frames, and does it smoothly:

| `--verify-span` | full sweeps | frames/FOV | focus error, mean | p95 |
|---:|---:|---:|---:|---:|
| 1.2 µm | 51.2% | 11.30 | 0.21 µm | 0.55 µm |
| 2.0 µm *(default)* | 44.2% | 10.14 | 0.30 µm | 0.60 µm |
| 3.0 µm | 34.9% | 8.86 | 0.35 µm | 0.70 µm |

A wider span accepts more predictions and interpolates the focus from further
away. At 0.55 µm of defocus this camera's blur is 0.3 px, so the default trades
accuracy nobody can see for one frame in ten.

### 2. Skipping background: 44% of the grid is never visited

Otsu on the ×32 thumbnail, opened and closed with a 3×3 kernel, then the
serpentine path is planned over the tissue bounding box and every FOV below 2%
coverage is dropped.

```
segmentation   otsu=208  tissue=20.8% of slide  bbox=(1152,0)-(5792,2784)  73 ms
plan           11 x 7 grid, 43 of 77 FOVs carry tissue (44.2% skipped)
```

44% fewer FOVs is 44% less of everything downstream — stage moves, exposures,
autofocus, stitching, pixels.

### 3. Stitching: 0.01 px mean registration error

Neighbouring FOVs overlap by 64 px. A 64×64 patch from each side of the overlap
band goes through Hann window → FFT → normalised cross-power spectrum → inverse
FFT → parabolic sub-pixel peak. The stage's positioning error is known exactly
(the synthetic stage records where it really went), so the recovered shift can be
scored rather than admired:

```
registration   65 overlaps, 64x64 patches: mean 0.01 px, median 0.01 px, p95 0.04 px, 0 over 1 px
mosaic         5008 x 3216 canvas, 43 tiles in 2 registration components
               placement error vs truth: mean 0.00 px, max 0.00 px
```

Two components, not one, because the slide has physically separate tissue
sections: tiles that share no overlap **cannot** be registered to each other, and
each island is placed relative to its own seed rather than being given an invented
position.

There is a second-order result hiding in the `--trust-map` run. Defocus the
tiles and registration degrades with them — mean 0.01 px → 0.86 px, p95 6.16 px,
10 of 65 pairs over a pixel, and the mosaic fragments into 8 components as the
correlation-peak gate rejects the links it cannot trust. **Focus quality
propagates into stitching quality**, which is not obvious until you can measure
both.

### 4. Flat-field correction

Gain estimated from a blank frame, applied to every acquired FOV:

```
flat-field     corner/centre 0.656 -> 1.000
```

### 5. The pyramid, and the resample loop that builds it

The output is a hand-written tiled pyramidal TIFF — classic little-endian TIFF,
one IFD per level, 256 px tiles, no libtiff. Verified by three readers: this
repo's own parser (unit test), `tifffile`, and **OpenSlide, which opens it as
`generic-tiff` with 6 levels** and serves `read_region` out of it.

```
pyramid        6 levels, 256 px tiles, 70.6 MB
               L0  5008 x  3216  20 x 13 tiles
               L1  2504 x  1608  10 x 7 tiles
               ...
               resample 35 ms (2.33 GB/s), write 40 ms
```

The whole pyramid is repeated 2×2 box downsamples, so that loop is worth writing
twice — once generically, once for the case that actually runs. Row pointers
instead of an index computation per sample, on 4096×4096×3:

```
downsample 2x   4096x4096x3
  generic NxN     106.0 ms   0.59 GB/s
  specialised      16.1 ms   3.91 GB/s   (6.60x)
```

A unit test asserts the fast path is **bit-identical** to the generic one, at
odd sizes and both channel counts — a faster resample that quietly rounds
differently is not a faster resample.

Other inner loops, same box:

```
focus metrics   192x192 ROI
  brenner         0.038 ms/frame   969.6 MPix/s
  var-laplacian   0.057 ms/frame   650.3 MPix/s
phase corr       64x64          0.397 ms/pair
phase corr      128x128         1.644 ms/pair
phase corr      256x256         7.376 ms/pair
```

### 6. Tile quality, in the loop, closing back onto the stage

A 61k-parameter CNN over 128×128 patches at native resolution — five classes:
sharp / blurred / folded / bubble / background. Trained on synthetic
degradations of clean tissue (labels are free), exported to ONNX, loaded by the
C++ scanner through ONNX Runtime, run on a 4×4 grid of patches per FOV.

Held out by *position*, not at random — the test set is the right quarter of the
slide, because patches from one location differ only by their degradation and a
random split would put the same tissue on both sides:

| class | precision | recall | F1 |
|---|---:|---:|---:|
| sharp | 0.970 | 0.899 | 0.933 |
| blurred | 0.997 | 0.841 | 0.912 |
| folded | 0.992 | 0.972 | **0.982** |
| bubble | 0.822 | 0.879 | 0.849 |
| background | 0.802 | 1.000 | 0.890 |
| **macro** | | | **0.913** |

Throughput: **1,795 tiles/s** (torch, CPU) and **11,103 tiles/s** (torch, GTX
1050); in the scan loop through ONNX Runtime on 4 CPU threads, **862–945
patches/s**, i.e. 688 patches of a 43-FOV slide classified in ~0.75 s.

Held-out F1 is the easy number. The one that matters is what it catches on the
scanner's *own* output, scored against the slide's ground-truth artefact
geometry (an oriented band for a fold, a disc for a bubble — not their bounding
boxes, since a diagonal band's bounding box is mostly clean tissue):

```
vs slide truth: folds 18/19 patches, bubbles 4/4 patches detected
```

And the loop closes. In the `--trust-map` run, where autofocus is deliberately
left unverified:

```
rescan         17 FOVs re-focused, 12 now pass (5 still flagged)
```

12 genuinely defocused FOVs were caught by the model and fixed by re-focusing
while the slide was still on the stage. The other 5 are **false positives**: the
same 5 FOVs are flagged in all three runs — including the one where every single
FOV got a full sweep — and their focus error there is 0.02–0.52 µm, which is
sharp by any measure this camera can express. They are sparse, low-contrast
tissue that the `blurred` class dislikes. 5 of 43 FOVs (12%) sent for an
unnecessary re-acquisition is the honest standing cost of this model.

---

## Three things that were wrong, and how they showed up

**BatchNorm gave the model a cue that does not exist at inference.** The first
classifier scored 0.94 accuracy during training and **0.39 in eval mode** — it
had learned "this patch is brighter than the others in this batch", a real
signal while training on shuffled balanced batches and a meaningless one when
the scanner hands it one tile. GroupNorm normalises inside each sample, so
training and inference compute the same function. Same architecture, same data:
macro-F1 0.98 (4 classes) instead of 0.52.

**The model had never seen empty glass.** Training patches were sampled at ≥85%
tissue coverage. The first in-loop run called **182 of 688** scanned patches
blurred and **147** of them bubbles: a patch of bare glass is flat and bright,
which is exactly what a defocused patch and the inside of a bubble look like.
Adding a `background` class — sampled from <35% coverage, with the same defocus
range as everything else — dropped that to 31 blurred and 24 bubble, and the
scanner now excludes background patches from a FOV's verdict entirely.

**The autofocus search window was in the wrong place.** It swept a fixed
±12 µm around zero. The focal surface tilts ~16 µm across a 6144 px slide, so at
the far corner the true focus was *outside the search range* and the best the
sweep could do was report the edge of its own window — a 1.87 µm error on a full
sweep, which is what led to it. The sweep is now centred on the best estimate
available (the focus map, or the previous FOV).

---

## Build and run

Needs a C++20 compiler and CMake ≥ 3.16. Nothing else: no libtiff, no OpenCV, no
libpng. Images move between C++ and Python as binary PNM, which Pillow reads.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

scripts/demo.sh                 # generate a slide, test, bench, and three scans
```

With the tile-quality model in the loop:

```bash
ORT=$(scripts/get_onnxruntime.sh 1.22.0)          # one zip from the ORT release page
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWSI_WITH_ONNX=ON -DORT_ROOT=$ORT
cmake --build build --parallel
build/wsi_scan --slide data/slide.ppm --truth data/slide_truth.json --out out \
               --model models/tile_quality.onnx --threads 4
```

Retrain the model (`torch`, `numpy`, `pillow`, `onnx`; ~20 s on a GTX 1050,
a couple of minutes on CPU):

```bash
python python/train_quality.py --slide data/slide.ppm --truth data/slide_truth.json --out models
```

Serve the scan (`fastapi`, `uvicorn`, `pillow`):

```bash
uvicorn python.service:app --port 8000
#  /                    a minimal tiled viewer
#  /info                levels and tile geometry
#  /tiles/{l}/{x}/{y}.png
#  /report              the scan report
#  /metrics             Prometheus counters, per level
```

The service reads the pyramid with ~70 lines of `struct.unpack` rather than a
TIFF library — the file the scanner writes is a plain tiled pyramid, so serving
one tile is one seek and one read.

**On real data**, convert any image (or a real `.svs`, if `openslide-python` is
installed) and scan it:

```bash
python python/make_slide.py --input CMU-1-Small-Region.svs --out data/real.ppm
build/wsi_scan --slide data/real.ppm --out out
```

Free samples: [OpenSlide test data](https://openslide.cs.cmu.edu/download/openslide-testdata/),
or CAMELYON16. You get the pipeline, the timings and the tile classification —
but not the error metrics, because a real slide does not come with the answers.

---

## Layout

| | |
|---|---|
| [include/wsi/](include/wsi/), [src/](src/) | the library: image, focus metrics, Otsu + flat-field, FFT + phase correlation, TIFF writer, camera/stage, planner + autofocus, ONNX classifier |
| [apps/scan_main.cpp](apps/scan_main.cpp) | the scanner, and the report it writes |
| [apps/makeslide.cpp](apps/makeslide.cpp) | the ground-truth slide generator |
| [apps/bench.cpp](apps/bench.cpp) | throughput of the three inner loops |
| [tests/test_main.cpp](tests/test_main.cpp) | 39 assertions, no framework |
| [python/train_quality.py](python/train_quality.py) | dataset synthesis, training, ONNX export |
| [python/service.py](python/service.py) | tile server, viewer, Prometheus metrics |
| [python/make_slide.py](python/make_slide.py) | real image / `.svs` → the scanner's input |
| [results/](results/) | the reports every number above came from |

The scanner talks to `ICamera` and `IStage` and nothing else. A real camera SDK
or a robot stage is one more implementation of two interfaces; the scan loop,
autofocus and stitching never learn where the pixels came from.

## What this is not

- **The slide is generated.** Procedural H&E-like tissue with nuclei, folds and
  bubbles. It exists so that every error can be scored; it is not a claim about
  real histology, and the tile-quality model is trained on *this generator's*
  idea of a fold.
- **The instrument is simulated**, including its time. "Simulated instrument
  time" is a cost model (stage settle + travel/speed, fixed exposure), not a
  measurement of hardware. Wall-clock timings are real.
- **TIFF tiles are uncompressed.** Real WSI files are JPEG- or JPEG2000-in-TIFF.
  Compression would be a codec dependency and would not change the layout, which
  is the part worth writing by hand.
- **No pybind11 layer.** The Python side reads the scanner's output files rather
  than calling into it; that is enough for the tile server and keeps the build to
  one compiler and one optional dependency.
- Single-threaded acquisition. The obvious next step is overlapping stage motion
  with the previous FOV's processing.
