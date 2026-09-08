"""A tile server over the pyramidal TIFF the scanner writes.

    uvicorn python.service:app --port 8000
    curl localhost:8000/info
    curl -o t.png localhost:8000/tiles/1/2/3.png

The TIFF reader here is ~70 lines and has no dependency on libtiff or
OpenSlide. That is the point: the file the scanner writes is a plain tiled
pyramid, so serving one tile means seeking to one offset and returning the
bytes. (OpenSlide opens the same file as `generic-tiff` -- see the README.)
"""

import io
import json
import os
import pathlib
import struct
import threading
import time

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

TIFF_PATH = os.environ.get("WSI_TIFF", "out/scan.tif")
REPORT_PATH = os.environ.get("WSI_REPORT", "out/report.json")

TAGS = {256: "width", 257: "height", 258: "bits", 259: "compression", 262: "photometric",
        277: "samples", 322: "tile_w", 323: "tile_h", 324: "offsets", 325: "counts"}


class TiffPyramid:
    """Classic little-endian TIFF, uncompressed tiles, one IFD per level."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.lock = threading.Lock()
        self.fh = open(self.path, "rb")
        magic = self.fh.read(8)
        if magic[:2] != b"II" or struct.unpack("<H", magic[2:4])[0] != 42:
            raise ValueError(f"{path}: not a little-endian classic TIFF")
        self.levels = []
        offset = struct.unpack("<I", magic[4:8])[0]
        while offset:
            ifd, offset = self._read_ifd(offset)
            self.levels.append(ifd)

    def _read_ifd(self, offset):
        self.fh.seek(offset)
        (count,) = struct.unpack("<H", self.fh.read(2))
        entries = {}
        for _ in range(count):
            tag, typ, n, value = struct.unpack("<HHII", self.fh.read(12))
            name = TAGS.get(tag)
            if name is None:
                continue
            if name in ("offsets", "counts") and n > 1:
                entries[name] = ("array", value, n)
            else:
                entries[name] = value
        (next_ifd,) = struct.unpack("<I", self.fh.read(4))

        for key in ("offsets", "counts"):
            v = entries.get(key)
            if isinstance(v, tuple):
                _, at, n = v
                self.fh.seek(at)
                entries[key] = list(struct.unpack(f"<{n}I", self.fh.read(4 * n)))
            else:
                entries[key] = [v]
        entries["tiles_x"] = (entries["width"] + entries["tile_w"] - 1) // entries["tile_w"]
        entries["tiles_y"] = (entries["height"] + entries["tile_h"] - 1) // entries["tile_h"]
        return entries, next_ifd

    def info(self):
        return {
            "path": str(self.path),
            "levels": [
                {"level": i, "width": L["width"], "height": L["height"],
                 "tile": L["tile_w"], "tiles_x": L["tiles_x"], "tiles_y": L["tiles_y"]}
                for i, L in enumerate(self.levels)
            ],
        }

    def tile_png(self, level, tx, ty):
        if not 0 <= level < len(self.levels):
            raise KeyError("level")
        L = self.levels[level]
        if not (0 <= tx < L["tiles_x"] and 0 <= ty < L["tiles_y"]):
            raise KeyError("tile")
        idx = ty * L["tiles_x"] + tx
        with self.lock:
            self.fh.seek(L["offsets"][idx])
            raw = self.fh.read(L["counts"][idx])
        mode = "RGB" if L.get("samples", 1) == 3 else "L"
        img = Image.frombytes(mode, (L["tile_w"], L["tile_h"]), raw)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


pyramid = TiffPyramid(TIFF_PATH)
app = FastAPI(title="wsi-scan-pipeline tile server")

metrics = {"requests": 0, "errors": 0, "bytes": 0, "seconds": 0.0,
           "per_level": {}}


@app.get("/info")
def info():
    return pyramid.info()


@app.get("/report")
def report():
    p = pathlib.Path(REPORT_PATH)
    if not p.exists():
        raise HTTPException(404, "no scan report; run wsi_scan first")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/tiles/{level}/{x}/{y}.png")
def tile(level: int, x: int, y: int):
    t0 = time.perf_counter()
    try:
        png = pyramid.tile_png(level, x, y)
    except KeyError as e:
        metrics["errors"] += 1
        raise HTTPException(404, f"no such {e.args[0]}")
    dt = time.perf_counter() - t0
    metrics["requests"] += 1
    metrics["bytes"] += len(png)
    metrics["seconds"] += dt
    lv = metrics["per_level"].setdefault(level, {"requests": 0, "seconds": 0.0})
    lv["requests"] += 1
    lv["seconds"] += dt
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/metrics")
def prometheus():
    lines = [
        "# HELP wsi_tile_requests_total Tiles served.",
        "# TYPE wsi_tile_requests_total counter",
        f"wsi_tile_requests_total {metrics['requests']}",
        "# HELP wsi_tile_errors_total Tile requests that named a tile that does not exist.",
        "# TYPE wsi_tile_errors_total counter",
        f"wsi_tile_errors_total {metrics['errors']}",
        "# HELP wsi_tile_bytes_total Bytes of PNG returned.",
        "# TYPE wsi_tile_bytes_total counter",
        f"wsi_tile_bytes_total {metrics['bytes']}",
        "# HELP wsi_tile_seconds_total Time spent reading and encoding tiles.",
        "# TYPE wsi_tile_seconds_total counter",
        f"wsi_tile_seconds_total {metrics['seconds']:.6f}",
        "# HELP wsi_pyramid_levels Levels in the open slide.",
        "# TYPE wsi_pyramid_levels gauge",
        f"wsi_pyramid_levels {len(pyramid.levels)}",
    ]
    for level, v in sorted(metrics["per_level"].items()):
        lines.append(f'wsi_tile_requests_by_level{{level="{level}"}} {v["requests"]}')
        lines.append(f'wsi_tile_seconds_by_level{{level="{level}"}} {v["seconds"]:.6f}')
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


VIEWER = """<!doctype html><title>wsi-scan-pipeline</title>
<style>body{font:14px system-ui;margin:16px;background:#111;color:#ddd}
#c{position:relative;overflow:auto;max-height:80vh;border:1px solid #333;background:#000}
img{position:absolute;display:block}button{margin-right:6px}</style>
<h3>wsi-scan-pipeline &mdash; <span id="lbl"></span></h3>
<div><button onclick="go(-1)">zoom in</button><button onclick="go(1)">zoom out</button>
<a href="/info" style="color:#6af">/info</a> &middot;
<a href="/report" style="color:#6af">/report</a> &middot;
<a href="/metrics" style="color:#6af">/metrics</a></div>
<div id="c"></div>
<script>
let info, level;
async function boot(){ info = await (await fetch('/info')).json();
  level = Math.min(2, info.levels.length-1); draw(); }
function go(d){ level = Math.max(0, Math.min(info.levels.length-1, level+d)); draw(); }
function draw(){ const L=info.levels[level], c=document.getElementById('c');
  c.innerHTML=''; c.style.width=L.width+'px'; c.style.height=L.height+'px';
  document.getElementById('lbl').textContent =
    `level ${level} of ${info.levels.length-1} — ${L.width}×${L.height}, ${L.tiles_x}×${L.tiles_y} tiles`;
  for(let ty=0;ty<L.tiles_y;ty++) for(let tx=0;tx<L.tiles_x;tx++){
    const im=new Image(); im.src=`/tiles/${level}/${tx}/${ty}.png`;
    im.style.left=(tx*L.tile)+'px'; im.style.top=(ty*L.tile)+'px'; c.appendChild(im);}}
boot();
</script>"""


@app.get("/", response_class=HTMLResponse)
def viewer():
    return VIEWER
