#include "wsi/tiff.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace wsi {
namespace {

constexpr uint16_t T_SUBFILE = 254, T_WIDTH = 256, T_LENGTH = 257, T_BITS = 258;
constexpr uint16_t T_COMPRESSION = 259, T_PHOTOMETRIC = 262, T_SAMPLES = 277;
constexpr uint16_t T_PLANAR = 284, T_TILE_W = 322, T_TILE_H = 323;
constexpr uint16_t T_TILE_OFFSETS = 324, T_TILE_COUNTS = 325;
constexpr uint16_t TYPE_SHORT = 3, TYPE_LONG = 4;

void put16(std::ostream& o, uint16_t v) { o.write(reinterpret_cast<const char*>(&v), 2); }
void put32(std::ostream& o, uint32_t v) { o.write(reinterpret_cast<const char*>(&v), 4); }

uint16_t get16(const std::vector<uint8_t>& b, size_t off) {
    return uint16_t(b[off] | (b[off + 1] << 8));
}
uint32_t get32(const std::vector<uint8_t>& b, size_t off) {
    return uint32_t(b[off]) | (uint32_t(b[off + 1]) << 8) | (uint32_t(b[off + 2]) << 16) |
           (uint32_t(b[off + 3]) << 24);
}

struct Entry {
    uint16_t tag, type;
    uint32_t count, value;
};

void write_entry(std::ostream& o, const Entry& e) {
    put16(o, e.tag);
    put16(o, e.type);
    put32(o, e.count);
    // A SHORT that fits inline sits in the low half of the 4-byte value field.
    put32(o, e.value);
}

// One pyramid level as it exists on disk.
struct LevelData {
    int w = 0, h = 0, ch = 0, tx = 0, ty = 0;
    std::vector<uint32_t> offsets, counts;
};

}  // namespace

PyramidStats write_pyramidal_tiff(const std::string& path, const Image& level0, int tile_size) {
    if (level0.empty()) throw std::runtime_error("write_pyramidal_tiff: empty level 0");
    if (tile_size <= 0 || (tile_size & (tile_size - 1)) != 0)
        throw std::runtime_error("tile size must be a power of two");

    PyramidStats st;
    st.tile_size = tile_size;
    const int ch = level0.ch;

    auto t0 = std::chrono::steady_clock::now();
    std::vector<Image> pyr;
    pyr.push_back(level0);
    uint64_t resample_bytes = 0;
    while (pyr.back().w > tile_size || pyr.back().h > tile_size) {
        const Image& prev = pyr.back();
        if (prev.w < 2 || prev.h < 2) break;
        resample_bytes += prev.bytes();            // read
        Image next = downsample2(prev);
        resample_bytes += next.bytes();            // write
        pyr.push_back(std::move(next));
        if (pyr.size() > 12) break;
    }
    auto t1 = std::chrono::steady_clock::now();
    st.resample_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    st.resample_gbs = st.resample_ms > 0
                          ? double(resample_bytes) / (st.resample_ms * 1e-3) / 1e9
                          : 0.0;

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot write " + path);

    // 8-byte header; the first-IFD pointer is patched once the IFDs are placed.
    f.write("II", 2);
    put16(f, 42);
    put32(f, 0);

    auto t2 = std::chrono::steady_clock::now();
    std::vector<LevelData> levels;
    std::vector<uint8_t> tile(size_t(tile_size) * tile_size * ch);
    for (const Image& img : pyr) {
        LevelData ld;
        ld.w = img.w;
        ld.h = img.h;
        ld.ch = ch;
        ld.tx = (img.w + tile_size - 1) / tile_size;
        ld.ty = (img.h + tile_size - 1) / tile_size;
        for (int ry = 0; ry < ld.ty; ++ry) {
            for (int rx = 0; rx < ld.tx; ++rx) {
                // Edge tiles are full-size and padded with glass white, which is
                // what the TIFF spec requires and what a viewer expects to see.
                std::fill(tile.begin(), tile.end(), uint8_t(255));
                for (int y = 0; y < tile_size; ++y) {
                    int sy = ry * tile_size + y;
                    if (sy >= img.h) break;
                    int sx = rx * tile_size;
                    int n = std::min(tile_size, img.w - sx);
                    if (n <= 0) break;
                    std::memcpy(&tile[size_t(y) * tile_size * ch], &img.px[img.idx(sx, sy)],
                                size_t(n) * ch);
                }
                ld.offsets.push_back(uint32_t(f.tellp()));
                ld.counts.push_back(uint32_t(tile.size()));
                f.write(reinterpret_cast<const char*>(tile.data()), std::streamsize(tile.size()));
            }
        }
        levels.push_back(std::move(ld));
    }

    // IFDs, in resolution order, each followed by its out-of-line arrays.
    uint32_t prev_next_ptr = 4;  // header slot for the first IFD
    for (size_t li = 0; li < levels.size(); ++li) {
        const LevelData& ld = levels[li];
        const uint32_t ntiles = uint32_t(ld.offsets.size());
        const bool bits_inline = ch <= 2;      // 2 bytes per sample value
        const bool arrays_inline = ntiles == 1;
        const uint16_t n_entries = 12;

        uint32_t ifd_off = uint32_t(f.tellp());
        uint32_t ifd_size = 2u + 12u * n_entries + 4u;
        uint32_t arr_off = ifd_off + ifd_size;
        uint32_t bits_off = arr_off;
        uint32_t off_off = bits_off + (bits_inline ? 0u : 2u * uint32_t(ch));
        uint32_t cnt_off = off_off + (arrays_inline ? 0u : 4u * ntiles);

        uint32_t bits_val = 0;
        if (bits_inline) {
            for (int c = 0; c < ch; ++c) bits_val |= uint32_t(8) << (16 * c);
        } else {
            bits_val = bits_off;
        }

        put16(f, n_entries);
        write_entry(f, {T_SUBFILE, TYPE_LONG, 1, li == 0 ? 0u : 1u});  // 1 = reduced resolution
        write_entry(f, {T_WIDTH, TYPE_LONG, 1, uint32_t(ld.w)});
        write_entry(f, {T_LENGTH, TYPE_LONG, 1, uint32_t(ld.h)});
        write_entry(f, {T_BITS, TYPE_SHORT, uint32_t(ch), bits_val});
        write_entry(f, {T_COMPRESSION, TYPE_SHORT, 1, 1});
        write_entry(f, {T_PHOTOMETRIC, TYPE_SHORT, 1, ch == 1 ? 1u : 2u});
        write_entry(f, {T_SAMPLES, TYPE_SHORT, 1, uint32_t(ch)});
        write_entry(f, {T_PLANAR, TYPE_SHORT, 1, 1});
        write_entry(f, {T_TILE_W, TYPE_LONG, 1, uint32_t(tile_size)});
        write_entry(f, {T_TILE_H, TYPE_LONG, 1, uint32_t(tile_size)});
        write_entry(f, {T_TILE_OFFSETS, TYPE_LONG, ntiles, arrays_inline ? ld.offsets[0] : off_off});
        write_entry(f, {T_TILE_COUNTS, TYPE_LONG, ntiles, arrays_inline ? ld.counts[0] : cnt_off});
        uint32_t next_ptr_pos = uint32_t(f.tellp());
        put32(f, 0);  // patched when the next IFD is written

        if (!bits_inline)
            for (int c = 0; c < ch; ++c) put16(f, 8);
        if (!arrays_inline) {
            for (uint32_t v : ld.offsets) put32(f, v);
            for (uint32_t v : ld.counts) put32(f, v);
        }

        f.flush();
        auto here = f.tellp();
        f.seekp(prev_next_ptr);
        put32(f, ifd_off);
        f.seekp(here);
        prev_next_ptr = next_ptr_pos;

        st.levels.push_back({ld.w, ld.h, ld.tx, ld.ty});
    }
    st.bytes_written = uint64_t(f.tellp());
    f.close();
    auto t3 = std::chrono::steady_clock::now();
    st.write_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    return st;
}

namespace {

struct Ifd {
    std::vector<Entry> entries;
    uint32_t next = 0;
    const Entry* find(uint16_t tag) const {
        for (const auto& e : entries)
            if (e.tag == tag) return &e;
        return nullptr;
    }
};

std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
}

std::vector<Ifd> parse_ifds(const std::vector<uint8_t>& b) {
    std::vector<Ifd> out;
    if (b.size() < 8 || b[0] != 'I' || b[1] != 'I' || get16(b, 2) != 42) return out;
    uint32_t off = get32(b, 4);
    while (off && off + 2 <= b.size()) {
        Ifd ifd;
        uint16_t n = get16(b, off);
        for (uint16_t i = 0; i < n; ++i) {
            size_t e = off + 2 + size_t(i) * 12;
            if (e + 12 > b.size()) return out;
            ifd.entries.push_back({get16(b, e), get16(b, e + 2), get32(b, e + 4), get32(b, e + 8)});
        }
        ifd.next = get32(b, off + 2 + size_t(n) * 12);
        out.push_back(std::move(ifd));
        off = out.back().next;
        if (out.size() > 32) break;
    }
    return out;
}

}  // namespace

TiffInfo read_tiff_info(const std::string& path) {
    TiffInfo info;
    auto b = slurp(path);
    auto ifds = parse_ifds(b);
    if (ifds.empty()) return info;
    for (const auto& ifd : ifds) {
        const Entry* w = ifd.find(T_WIDTH);
        const Entry* h = ifd.find(T_LENGTH);
        const Entry* tw = ifd.find(T_TILE_W);
        const Entry* th = ifd.find(T_TILE_H);
        const Entry* sp = ifd.find(T_SAMPLES);
        if (!w || !h || !tw || !th) return info;
        info.tile_size = int(tw->value);
        info.channels = sp ? int(sp->value) : 1;
        int tx = (int(w->value) + int(tw->value) - 1) / int(tw->value);
        int ty = (int(h->value) + int(th->value) - 1) / int(th->value);
        info.levels.push_back({int(w->value), int(h->value), tx, ty});
        (void)th;
    }
    info.ok = true;
    return info;
}

Image read_tiff_tile(const std::string& path, int level, int tx, int ty) {
    auto b = slurp(path);
    auto ifds = parse_ifds(b);
    if (level < 0 || level >= int(ifds.size())) throw std::runtime_error("no such TIFF level");
    const Ifd& ifd = ifds[size_t(level)];
    const Entry* w = ifd.find(T_WIDTH);
    const Entry* tw = ifd.find(T_TILE_W);
    const Entry* sp = ifd.find(T_SAMPLES);
    const Entry* to = ifd.find(T_TILE_OFFSETS);
    const Entry* tc = ifd.find(T_TILE_COUNTS);
    if (!w || !tw || !to || !tc) throw std::runtime_error("malformed TIFF directory");

    int tile = int(tw->value);
    int ch = sp ? int(sp->value) : 1;
    int tiles_x = (int(w->value) + tile - 1) / tile;
    uint32_t index = uint32_t(ty) * uint32_t(tiles_x) + uint32_t(tx);
    if (index >= to->count) throw std::runtime_error("tile index out of range");

    uint32_t off = to->count == 1 ? to->value : get32(b, to->value + 4u * index);
    uint32_t len = tc->count == 1 ? tc->value : get32(b, tc->value + 4u * index);
    Image img(tile, tile, ch);
    if (off + len > b.size() || len != img.px.size()) throw std::runtime_error("bad tile extent");
    std::memcpy(img.px.data(), &b[off], len);
    return img;
}

}  // namespace wsi
