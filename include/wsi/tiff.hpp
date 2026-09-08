// A tiled, pyramidal TIFF writer (and just enough of a reader to test it).
//
// Written by hand rather than pulled from libtiff: the whole point of the file
// is that a viewer can fetch one 256x256 tile at one zoom level without
// touching the rest of the slide, and that is a property of the layout, not of
// the library. Classic (32-bit) TIFF, little endian, uncompressed.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "wsi/image.hpp"

namespace wsi {

struct TiffLevel {
    int w = 0, h = 0;
    int tiles_x = 0, tiles_y = 0;
};

struct PyramidStats {
    std::vector<TiffLevel> levels;
    int tile_size = 0;
    uint64_t bytes_written = 0;
    double resample_ms = 0;  // time spent building levels 1..n
    double write_ms = 0;     // time spent laying tiles into the file
    double resample_gbs = 0; // bytes touched by the resampler / resample_ms
};

// Builds levels by repeated 2x2 box downsample until the largest dimension
// fits in a single tile, then writes one TIFF directory per level.
PyramidStats write_pyramidal_tiff(const std::string& path, const Image& level0,
                                  int tile_size = 256);

// Reader used by the tests: parses the IFD chain back out of the file.
struct TiffInfo {
    std::vector<TiffLevel> levels;
    int tile_size = 0;
    int channels = 0;
    bool ok = false;
};
TiffInfo read_tiff_info(const std::string& path);
Image read_tiff_tile(const std::string& path, int level, int tx, int ty);

}  // namespace wsi
