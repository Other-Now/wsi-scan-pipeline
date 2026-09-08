"""Turn a real image into a slide the scanner can scan.

The pipeline reads P6 PNM, which every tool can write and nothing needs a
library to parse. This converts anything Pillow can open -- and, if OpenSlide
is installed, a real .svs or .ndpi whole-slide image:

    python python/make_slide.py --input CMU-1-Small-Region.svs --out data/real.ppm
    python python/make_slide.py --input tissue.png --out data/real.ppm --max-side 8192

Free .svs samples: https://openslide.cs.cmu.edu/download/openslide-testdata/
CAMELYON16 works too. Note that wsi_scan's error metrics (focus, registration,
placement) are all *against ground truth* and so are only meaningful on the
generated slide -- on a real one you get the pipeline, the timings and the
tile-quality classification, but nothing to score them against.
"""

import argparse
import pathlib

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def open_any(path, max_side):
    p = pathlib.Path(path)
    if p.suffix.lower() in (".svs", ".ndpi", ".mrxs", ".scn", ".vms"):
        try:
            import openslide
        except ImportError:
            raise SystemExit("reading a whole-slide format needs: pip install openslide-python "
                             "openslide-bin")
        slide = openslide.OpenSlide(str(p))
        # Smallest level that still has more pixels than we want, then downscale.
        level = slide.get_best_level_for_downsample(max(slide.dimensions) / max_side)
        img = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB")
        print(f"openslide: {slide.dimensions} -> level {level} {img.size}")
        slide.close()
        return img
    return Image.open(p).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data/real_slide.ppm")
    ap.add_argument("--max-side", type=int, default=8192)
    args = ap.parse_args()

    img = open_any(args.input, args.max_side)
    if max(img.size) > args.max_side:
        scale = args.max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    # Even dimensions keep every pyramid level an exact 2x of the one above it.
    img = img.crop((0, 0, img.width - img.width % 2, img.height - img.height % 2))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PPM")
    print(f"wrote {out} ({img.width}x{img.height}, {out.stat().st_size / 1e6:.1f} MB)")
    print("no ground truth for a real slide: run wsi_scan without --truth")


if __name__ == "__main__":
    main()
