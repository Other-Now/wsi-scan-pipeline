// The scanner: segment -> plan -> autofocus + acquire -> flat-field -> stitch
// -> pyramidal TIFF -> (optional) tile-quality classification and rescan.
//
// Every stage writes its numbers into out/report.json. The numbers that matter
// are errors against a known truth, which is the whole reason the slide is
// synthetic: the camera knows where the stage really landed and where the focal
// plane really is, and the scanner is scored against that.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "wsi/camera.hpp"
#include "wsi/classifier.hpp"
#include "wsi/focus.hpp"
#include "wsi/registration.hpp"
#include "wsi/scan.hpp"
#include "wsi/segment.hpp"
#include "wsi/slide.hpp"
#include "wsi/tiff.hpp"

using namespace wsi;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * double(v.size() - 1);
    size_t lo = size_t(idx);
    size_t hi = std::min(v.size() - 1, lo + 1);
    double frac = idx - double(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

double mean(const std::vector<double>& v) {
    return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
}

int largest_pow2_at_most(int n) {
    int p = 1;
    while (p * 2 <= n) p *= 2;
    return p;
}

// Ground-truth artefact boxes, as written by wsi_makeslide. Small enough a
// hand-rolled scan beats taking on a JSON dependency.
std::vector<Artifact> read_truth(const std::string& path) {
    std::vector<Artifact> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t pos = 0;
    while ((pos = text.find("{\"kind\"", pos)) != std::string::npos) {
        size_t end = text.find('}', pos);
        if (end == std::string::npos) break;
        std::string rec = text.substr(pos, end - pos);
        Artifact a;
        size_t k = rec.find("\"kind\": \"");
        if (k != std::string::npos) {
            k += 9;
            a.kind = rec.substr(k, rec.find('"', k) - k);
        }
        auto num = [&](const char* key) {
            size_t p = rec.find(key);
            return p == std::string::npos ? 0.0 : std::atof(rec.c_str() + p + std::strlen(key));
        };
        a.x = int(num("\"x\": "));
        a.y = int(num("\"y\": "));
        a.w = int(num("\"w\": "));
        a.h = int(num("\"h\": "));
        a.cx = num("\"cx\": ");
        a.cy = num("\"cy\": ");
        a.angle = num("\"angle\": ");
        a.half_w = num("\"half_w\": ");
        a.length = num("\"length\": ");
        out.push_back(a);
        pos = end;
    }
    return out;
}

// How much of a patch actually sits on the artefact, sampled on an 8x8 grid
// against the real shape (an oriented band, or a disc) rather than its bounding
// box. A diagonal fold's bounding box is mostly clean tissue, and scoring
// against it would count patches the model was never wrong about.
double artifact_coverage(int x, int y, int size, const Artifact& a) {
    const int n = 8;
    int inside = 0;
    const double ca = std::cos(a.angle), sa = std::sin(a.angle);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            double px = x + (i + 0.5) * size / n, py = y + (j + 0.5) * size / n;
            double dx = px - a.cx, dy = py - a.cy;
            if (a.kind == "bubble") {
                if (std::hypot(dx, dy) <= a.half_w) ++inside;
            } else {
                double u = dx * ca + dy * sa;      // across the band
                double v = -dx * sa + dy * ca;     // along it
                if (std::abs(u) <= a.half_w + 11.0 && std::abs(v) <= a.length / 2) ++inside;
            }
        }
    }
    return double(inside) / double(n * n);
}

struct Acquired {
    int plan_index = 0;
    int col = 0, row = 0;
    int x = 0, y = 0;
    Image img;
    double z = 0, af_error_um = 0, focus_score = 0;
    int af_frames = 0;
    bool full_sweep = true;
    int est_jx = 0, est_jy = 0;  // estimated stage error, from registration
    bool placed = false;
    int cls = -1;
    float cls_prob = 0.f;
    bool rescanned = false;
};

struct Args {
    std::string slide = "slide.ppm";
    std::string truth = "slide_truth.json";
    std::string out = "out";
    std::string model;
    int fov = 512, overlap = 64, thumb_scale = 32, tile = 256;
    int af_roi = 192;
    bool use_map = true;
    bool verify_map = true;
    bool save_preview = true;
    int threads = 1;
    double verify_span = 2.0;
    double min_peak = 12.0;  // correlation peak below this: do not trust the link
    FocusMap::Mode map_mode = FocusMap::Mode::PlaneLocal;
    FocusMetric metric = FocusMetric::Brenner;
};

void usage() {
    std::printf(
        "usage: wsi_scan [--slide slide.ppm] [--truth slide_truth.json] [--out out]\n"
        "                [--fov 512] [--overlap 64] [--thumb-scale 32] [--tile 256]\n"
        "                [--af-roi 192] [--metric brenner|varlap] [--no-map]\n"
        "                [--map plane|local] [--verify-span 2.0] [--min-peak 12] [--trust-map]\n"
        "                [--model quality.onnx] [--threads 1] [--no-preview]\n");
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--slide") args.slide = next();
        else if (a == "--truth") args.truth = next();
        else if (a == "--out") args.out = next();
        else if (a == "--model") args.model = next();
        else if (a == "--fov") args.fov = std::atoi(next().c_str());
        else if (a == "--overlap") args.overlap = std::atoi(next().c_str());
        else if (a == "--thumb-scale") args.thumb_scale = std::atoi(next().c_str());
        else if (a == "--tile") args.tile = std::atoi(next().c_str());
        else if (a == "--af-roi") args.af_roi = std::atoi(next().c_str());
        else if (a == "--threads") args.threads = std::atoi(next().c_str());
        else if (a == "--metric") args.metric = next() == "varlap" ? FocusMetric::VarLaplacian
                                                                  : FocusMetric::Brenner;
        else if (a == "--verify-span") args.verify_span = std::atof(next().c_str());
        else if (a == "--min-peak") args.min_peak = std::atof(next().c_str());
        else if (a == "--map") args.map_mode = next() == "plane" ? FocusMap::Mode::Plane
                                                                : FocusMap::Mode::PlaneLocal;
        else if (a == "--no-map") args.use_map = false;
        else if (a == "--trust-map") args.verify_map = false;
        else if (a == "--no-preview") args.save_preview = false;
        else { usage(); return a == "--help" ? 0 : 2; }
    }

    auto t_start = Clock::now();
    Image slide;
    try {
        slide = read_pnm(args.slide);
    } catch (const std::exception& e) {
        std::printf("error: %s\n(run wsi_makeslide first)\n", e.what());
        return 1;
    }
    std::printf("slide          %d x %d (%.1f MB)\n", slide.w, slide.h,
                double(slide.bytes()) / 1e6);

    // ---- 1. tissue segmentation ------------------------------------------
    auto t0 = Clock::now();
    TissueSeg seg = segment_tissue(slide, args.thumb_scale);
    double seg_ms = ms_since(t0);
    std::printf("segmentation   otsu=%d  tissue=%.1f%% of slide  bbox=(%d,%d)-(%d,%d)  %.0f ms\n",
                seg.threshold, 100.0 * seg.tissue_frac, seg.x0, seg.y0, seg.x1, seg.y1, seg_ms);

    // ---- 2. plan ----------------------------------------------------------
    ScanPlan plan = plan_serpentine(seg, slide.w, slide.h, args.fov, args.fov, args.overlap);
    double skipped_frac = plan.total ? 1.0 - double(plan.to_acquire) / double(plan.total) : 0.0;
    std::printf("plan           %d x %d grid, %d of %d FOVs carry tissue (%.1f%% skipped)\n",
                plan.cols, plan.rows, plan.to_acquire, plan.total, 100.0 * skipped_frac);

    // ---- 3. instrument ----------------------------------------------------
    OpticsParams optics;
    optics.fov_w = optics.fov_h = args.fov;
    // Anchor the focus anomaly on a FOV that is actually going to be acquired,
    // otherwise it sits on empty glass and never tests anything.
    {
        std::vector<const ScanTile*> live;
        for (const ScanTile& t : plan.tiles)
            if (t.acquire) live.push_back(&t);
        if (!live.empty()) {
            const ScanTile* t = live[live.size() * 3 / 5];
            optics.anomaly_x = t->x + args.fov / 2;
            optics.anomaly_y = t->y + args.fov / 2;
            optics.anomaly_r = 700;
            optics.anomaly_dz = 6.5;
        }
    }

    SyntheticStage stage;
    SyntheticCamera cam(slide, stage, optics);

    // ---- 4. flat-field calibration ---------------------------------------
    t0 = Clock::now();
    Image blank = cam.blank_frame();
    FlatField ff = estimate_flatfield(blank);
    Image blank_corr = blank;
    apply_flatfield(blank_corr, ff);
    double ff_before = double(blank.at(2, 2, 0)) / double(blank.at(args.fov / 2, args.fov / 2, 0));
    double ff_after =
        double(blank_corr.at(2, 2, 0)) / double(blank_corr.at(args.fov / 2, args.fov / 2, 0));
    double ff_ms = ms_since(t0);
    std::printf("flat-field     corner/centre %.3f -> %.3f  (%.0f ms)\n", ff_before, ff_after,
                ff_ms);

    // ---- 5. acquisition ---------------------------------------------------
    AfConfig af;
    af.roi = args.af_roi;
    af.metric = args.metric;
    af.verify_span = args.verify_span;
    af.verify = args.verify_map;

    FocusMap map;
    map.set_mode(args.map_mode);
    map.set_local_radius(1.5 * args.fov);
    std::vector<Acquired> acq;
    acq.reserve(size_t(plan.to_acquire));
    std::vector<double> accepted_scores;
    std::vector<double> af_errors, af_errors_mapped, af_errors_swept;
    int full_sweeps = 0, mapped = 0, af_frames = 0;
    double af_time_ms = 0;
    double last_z = 0.0;

    t0 = Clock::now();
    for (size_t i = 0; i < plan.tiles.size(); ++i) {
        const ScanTile& t = plan.tiles[i];
        if (!t.acquire) continue;

        stage.move_xy(t.x, t.y);
        auto taf = Clock::now();
        double expected = accepted_scores.empty() ? 0.0 : percentile(accepted_scores, 0.5);
        // Before the map exists, the previous tile is the best guess there is;
        // neighbouring FOVs are 448 px apart and the surface barely moves.
        AfResult r = (args.use_map && map.ready())
                         ? autofocus_with_map(stage, cam, map, af, expected)
                         : autofocus_sweep(stage, cam, af, last_z);
        last_z = r.z;
        af_time_ms += ms_since(taf);
        af_frames += r.frames;
        if (r.full_sweep) {
            ++full_sweeps;
            map.observe(double(t.x), double(t.y), r.z);
        } else {
            ++mapped;
        }
        accepted_scores.push_back(r.score);

        Acquired a;
        a.plan_index = int(i);
        a.col = t.col;
        a.row = t.row;
        a.x = t.x;
        a.y = t.y;
        a.z = r.z;
        a.af_frames = r.frames;
        a.full_sweep = r.full_sweep;
        a.focus_score = r.score;
        a.af_error_um = std::abs(r.z - cam.true_focus_z(t.x, t.y));
        af_errors.push_back(a.af_error_um);
        (r.full_sweep ? af_errors_swept : af_errors_mapped).push_back(a.af_error_um);

        a.img = cam.grab();
        apply_flatfield(a.img, ff);
        acq.push_back(std::move(a));
    }
    double acquire_ms = ms_since(t0);
    double pct_full = acq.empty() ? 0.0 : 100.0 * double(full_sweeps) / double(acq.size());
    std::printf("autofocus      %zu tiles: %d full sweeps (%.1f%%), %d from the map\n", acq.size(),
                full_sweeps, pct_full, mapped);
    std::printf("               %d AF frames, %.2f per tile, %.0f ms search (%.1f ms/tile)\n",
                af_frames, acq.empty() ? 0.0 : double(af_frames) / double(acq.size()), af_time_ms,
                acq.empty() ? 0.0 : af_time_ms / double(acq.size()));
    std::printf("               focus error vs truth: mean %.2f um, p95 %.2f um  (mapped %.2f, swept %.2f)\n",
                mean(af_errors), percentile(af_errors, 0.95), mean(af_errors_mapped),
                mean(af_errors_swept));
    std::printf("               map: %zu observations, plane residual %.2f um\n", map.size(),
                map.rms_residual());
    std::printf("acquisition    %.0f ms wall, simulated instrument time %.0f ms "
                "(stage %.0f + exposure %.0f)\n",
                acquire_ms, stage.stats().move_time_ms + cam.stats().exposure_time_ms,
                stage.stats().move_time_ms, cam.stats().exposure_time_ms);

    // ---- 6. registration --------------------------------------------------
    // Index acquired tiles by grid position so neighbours are cheap to find.
    std::map<std::pair<int, int>, size_t> at;
    for (size_t i = 0; i < acq.size(); ++i) at[{acq[i].col, acq[i].row}] = i;

    const int patch = largest_pow2_at_most(std::min(args.overlap, args.fov / 2));
    std::vector<double> reg_err, reg_peak;
    int pairs = 0, pair_failures = 0, weak_links = 0;

    t0 = Clock::now();
    struct Link {
        size_t from, to;
        int dx, dy;  // measured stage error of `to` relative to `from`
    };
    std::vector<Link> links;

    for (size_t i = 0; i < acq.size(); ++i) {
        const Acquired& a = acq[i];
        // Left neighbour (horizontal overlap band) and upper neighbour.
        for (int dir = 0; dir < 2; ++dir) {
            auto it = at.find({a.col - (dir == 0 ? 1 : 0), a.row - (dir == 0 ? 0 : 1)});
            if (it == at.end()) continue;
            const Acquired& b = acq[it->second];  // the earlier tile

            Image pa, pb;
            if (dir == 0) {
                int ox = plan.step_x;                    // overlap band inside the left tile
                int oy = (args.fov - patch) / 2;
                pa = crop(to_gray(b.img), ox + (args.overlap - patch) / 2, oy, patch, patch);
                pb = crop(to_gray(a.img), (args.overlap - patch) / 2, oy, patch, patch);
            } else {
                int oy = plan.step_y;
                int ox = (args.fov - patch) / 2;
                pa = crop(to_gray(b.img), ox, oy + (args.overlap - patch) / 2, patch, patch);
                pb = crop(to_gray(a.img), ox, (args.overlap - patch) / 2, patch, patch);
            }
            Shift s = phase_correlate(pa, pb);
            if (!s.ok) continue;

            auto [jbx, jby] = cam.true_offset(b.x, b.y);
            auto [jax, jay] = cam.true_offset(a.x, a.y);
            double truth_dx = double(jax - jbx), truth_dy = double(jay - jby);
            double err = std::hypot(double(s.dx) - truth_dx, double(s.dy) - truth_dy);
            reg_err.push_back(err);
            reg_peak.push_back(s.peak);
            ++pairs;
            if (err > 1.0) ++pair_failures;

            // A flat correlation surface means the overlap band had nothing to
            // register on -- background, or a bubble. Measuring it is fine;
            // building the mosaic on it is not.
            if (s.peak < args.min_peak) {
                ++weak_links;
                continue;
            }
            links.push_back({it->second, i, int(std::lround(s.dx)), int(std::lround(s.dy))});
        }
    }
    double reg_ms = ms_since(t0);
    std::printf("registration   %d overlaps, %dx%d patches: mean %.2f px, median %.2f px, "
                "p95 %.2f px, %d over 1 px\n",
                pairs, patch, patch, mean(reg_err), percentile(reg_err, 0.5),
                percentile(reg_err, 0.95), pair_failures);
    std::printf("               peak: median %.0f, p5 %.0f  -- %d links rejected below %.0f\n",
                percentile(reg_peak, 0.5), percentile(reg_peak, 0.05), weak_links, args.min_peak);
    std::printf("               %.0f ms total, %.2f ms per overlap\n", reg_ms,
                pairs ? reg_ms / pairs : 0.0);

    // ---- 7. mosaic --------------------------------------------------------
    // Chain the measured pairwise offsets outward from the first acquired tile,
    // whose own stage error is unknowable from the images alone.
    t0 = Clock::now();
    // Breadth-first over the undirected link graph. Tissue arrives in separate
    // sections, so the graph can have several components; a component that
    // never touches the seed genuinely cannot be registered to it, and is left
    // where the stage said it was rather than being quietly invented.
    std::vector<std::vector<std::pair<size_t, std::pair<int, int>>>> adj(acq.size());
    for (const Link& l : links) {
        adj[l.from].push_back({l.to, {l.dx, l.dy}});
        adj[l.to].push_back({l.from, {-l.dx, -l.dy}});
    }
    std::vector<int> component(acq.size(), -1);
    std::vector<size_t> seed_of;  // component id -> its seed tile
    for (size_t start = 0; start < acq.size(); ++start) {
        if (acq[start].placed) continue;
        int cid = int(seed_of.size());
        seed_of.push_back(start);
        std::vector<size_t> queue{start};
        acq[start].placed = true;
        component[start] = cid;
        for (size_t qi = 0; qi < queue.size(); ++qi) {
            size_t u = queue[qi];
            for (const auto& [v, d] : adj[u]) {
                if (acq[v].placed) continue;
                acq[v].est_jx = acq[u].est_jx + d.first;
                acq[v].est_jy = acq[u].est_jy + d.second;
                acq[v].placed = true;
                component[v] = cid;
                queue.push_back(v);
            }
        }
    }
    const int components = int(seed_of.size());

    // Placement is only defined relative to a component's own seed tile: the
    // absolute stage error of that first tile is not observable from images.
    std::vector<double> place_err;
    for (size_t i = 0; i < acq.size(); ++i) {
        const Acquired& a = acq[i];
        const Acquired& seed = acq[seed_of[size_t(component[i])]];
        auto [j0x, j0y] = cam.true_offset(seed.x, seed.y);
        auto [jx, jy] = cam.true_offset(a.x, a.y);
        place_err.push_back(
            std::hypot(double(a.est_jx) - double(jx - j0x), double(a.est_jy) - double(jy - j0y)));
    }

    const int margin = 2 * int(optics.xy_jitter_px) + 2;
    int canvas_x0 = seg.x0 - margin, canvas_y0 = seg.y0 - margin;
    int canvas_w = (plan.cols - 1) * plan.step_x + args.fov + 2 * margin;
    int canvas_h = (plan.rows - 1) * plan.step_y + args.fov + 2 * margin;
    Image canvas(canvas_w, canvas_h, 3);
    std::fill(canvas.px.begin(), canvas.px.end(), uint8_t(245));
    std::vector<uint8_t> written(size_t(canvas_w) * canvas_h, 0);

    for (const Acquired& a : acq) {
        int px = a.x + a.est_jx - canvas_x0;
        int py = a.y + a.est_jy - canvas_y0;
        for (int y = 0; y < args.fov; ++y) {
            int cy = py + y;
            if (cy < 0 || cy >= canvas_h) continue;
            for (int x = 0; x < args.fov; ++x) {
                int cx = px + x;
                if (cx < 0 || cx >= canvas_w) continue;
                size_t wi = size_t(cy) * canvas_w + cx;
                for (int c = 0; c < 3; ++c) {
                    uint8_t v = a.img.at(x, y, c);
                    canvas.at(cx, cy, c) =
                        written[wi] ? uint8_t((int(canvas.at(cx, cy, c)) + int(v)) / 2) : v;
                }
                written[wi] = 1;
            }
        }
    }
    double mosaic_ms = ms_since(t0);
    std::printf("mosaic         %d x %d canvas, %zu tiles in %d registration component%s\n",
                canvas_w, canvas_h, acq.size(), components, components == 1 ? "" : "s");
    std::printf("               placement error vs truth: mean %.2f px, max %.2f px  (%.0f ms)\n",
                mean(place_err),
                place_err.empty() ? 0.0 : *std::max_element(place_err.begin(), place_err.end()),
                mosaic_ms);

    // ---- 8. pyramidal tiled TIFF -----------------------------------------
    std::string tif_path = args.out + "/scan.tif";
    PyramidStats pyr = write_pyramidal_tiff(tif_path, canvas, args.tile);
    std::printf("pyramid        %zu levels, %d px tiles, %.1f MB\n", pyr.levels.size(),
                pyr.tile_size, double(pyr.bytes_written) / 1e6);
    for (size_t i = 0; i < pyr.levels.size(); ++i)
        std::printf("               L%zu %5d x %5d  %d x %d tiles\n", i, pyr.levels[i].w,
                    pyr.levels[i].h, pyr.levels[i].tiles_x, pyr.levels[i].tiles_y);
    std::printf("               resample %.0f ms (%.2f GB/s), write %.0f ms\n", pyr.resample_ms,
                pyr.resample_gbs, pyr.write_ms);

    // ---- 9. tile quality, and a rescan of what it flags -------------------
    std::string qc_status = "disabled";
    double qc_tiles_per_s = 0;
    int rescanned = 0, rescan_fixed = 0, flagged_fovs = 0, patch_count = 0;
    std::map<std::string, int> class_counts;
    int fold_hits = 0, fold_truth = 0, bubble_hits = 0, bubble_truth = 0;
    double qc_ms = 0;

    if (!args.model.empty()) {
        std::string err;
        auto clf = make_onnx_classifier(args.model, args.threads, &err);
        if (!clf) {
            qc_status = "unavailable: " + err;
            std::printf("quality        %s\n", qc_status.c_str());
        } else {
            const int psize = clf->input_size();
            const int grid = std::max(1, args.fov / psize);
            const auto& names = clf->class_names();
            int blurred_idx = -1, background_idx = -1;
            for (size_t i = 0; i < names.size(); ++i) {
                if (names[i] == "blurred") blurred_idx = int(i);
                if (names[i] == "background") background_idx = int(i);
            }
            // A FOV is worth re-acquiring when a quarter of its *tissue* is out
            // of focus, and there is enough tissue for that to mean something.
            auto is_bad = [&](int blurred, int tissue) {
                return tissue >= 4 && blurred * 4 >= tissue;
            };

            // Each FOV is cut into a grid of model-sized patches at native
            // resolution. Squeezing a whole 512 px FOV into a 128 px model
            // input would throw away the very high-frequency content the model
            // is being asked to judge -- an out-of-focus tile downsampled 4x
            // looks a lot like a sharp one.
            struct Patch {
                size_t tile;
                int px, py;
            };
            std::vector<Patch> patches;
            std::vector<Image> images;
            auto cut = [&](size_t ti, std::vector<Patch>& ps, std::vector<Image>& ims) {
                const Acquired& a = acq[ti];
                for (int gy = 0; gy + psize <= args.fov; gy += psize)
                    for (int gx = 0; gx + psize <= args.fov; gx += psize) {
                        ps.push_back({ti, gx, gy});
                        ims.push_back(crop(a.img, gx, gy, psize, psize));
                    }
            };
            for (size_t i = 0; i < acq.size(); ++i) cut(i, patches, images);
            patch_count = int(images.size());

            std::vector<TileQuality> qs;
            qs.reserve(images.size());
            t0 = Clock::now();
            for (size_t i = 0; i < images.size(); i += 256) {
                std::vector<Image> chunk(images.begin() + long(i),
                                         images.begin() + long(std::min(images.size(), i + 256)));
                auto part = clf->classify(chunk);
                qs.insert(qs.end(), part.begin(), part.end());
            }
            qc_ms = ms_since(t0);
            qc_tiles_per_s = qc_ms > 0 ? double(images.size()) / (qc_ms * 1e-3) : 0;
            qc_status = "ok";

            // Background patches are excluded from the verdict: a FOV at the rim
            // of a section is mostly glass, and glass is not evidence of
            // anything being out of focus.
            std::vector<int> blurred_in_tile(acq.size(), 0), tissue_in_tile(acq.size(), 0);
            for (size_t i = 0; i < qs.size(); ++i) {
                class_counts[names[size_t(qs[i].cls)]]++;
                if (qs[i].cls == background_idx) continue;
                tissue_in_tile[patches[i].tile]++;
                if (qs[i].cls == blurred_idx) blurred_in_tile[patches[i].tile]++;
            }
            (void)grid;

            std::printf("quality        %d patches of %dx%d from %zu FOVs in %.0f ms "
                        "(%.0f patches/s, %d thread%s)\n",
                        patch_count, psize, psize, acq.size(), qc_ms, qc_tiles_per_s, args.threads,
                        args.threads == 1 ? "" : "s");
            std::printf("               ");
            for (const auto& [name, n] : class_counts) std::printf("%s=%d  ", name.c_str(), n);
            std::printf("\n");

            // Score fold and bubble detection against the generator's truth,
            // patch by patch, since that is the area the model actually saw.
            std::vector<Artifact> truth = read_truth(args.truth);
            if (!truth.empty()) {
                for (size_t i = 0; i < qs.size(); ++i) {
                    const Acquired& a = acq[patches[i].tile];
                    int gx = a.x + patches[i].px, gy = a.y + patches[i].py;
                    for (const Artifact& art : truth) {
                        if (artifact_coverage(gx, gy, psize, art) < 0.25) continue;
                        const std::string& pred = names[size_t(qs[i].cls)];
                        if (art.kind == "fold") {
                            ++fold_truth;
                            if (pred == "folded") ++fold_hits;
                        } else {
                            ++bubble_truth;
                            if (pred == "bubble") ++bubble_hits;
                        }
                        break;
                    }
                }
                std::printf("               vs slide truth: folds %d/%d patches, bubbles %d/%d "
                            "patches detected\n",
                            fold_hits, fold_truth, bubble_hits, bubble_truth);
            }

            // The point of running the model in-loop: a FOV the model calls
            // blurred is re-focused the expensive way and re-acquired while the
            // slide is still on the stage.
            std::vector<size_t> flagged;
            for (size_t i = 0; i < acq.size(); ++i)
                if (is_bad(blurred_in_tile[i], tissue_in_tile[i])) flagged.push_back(i);
            flagged_fovs = int(flagged.size());

            if (!flagged.empty()) {
                std::vector<Patch> rp;
                std::vector<Image> ri;
                for (size_t ti : flagged) {
                    Acquired& a = acq[ti];
                    stage.move_xy(a.x, a.y);
                    AfResult r = autofocus_sweep(stage, cam, af, a.z);
                    a.z = r.z;
                    a.rescanned = true;
                    a.af_error_um = std::abs(r.z - cam.true_focus_z(a.x, a.y));
                    a.img = cam.grab();
                    apply_flatfield(a.img, ff);
                    ++rescanned;
                    cut(ti, rp, ri);
                }
                std::vector<TileQuality> after;
                for (size_t i = 0; i < ri.size(); i += 256) {
                    std::vector<Image> chunk(ri.begin() + long(i),
                                             ri.begin() + long(std::min(ri.size(), i + 256)));
                    auto part = clf->classify(chunk);
                    after.insert(after.end(), part.begin(), part.end());
                }
                std::map<size_t, int> still_blurred, still_tissue;
                for (size_t i = 0; i < after.size(); ++i) {
                    if (after[i].cls == background_idx) continue;
                    still_tissue[rp[i].tile]++;
                    if (after[i].cls == blurred_idx) still_blurred[rp[i].tile]++;
                }
                for (size_t ti : flagged)
                    if (!is_bad(still_blurred[ti], still_tissue[ti])) ++rescan_fixed;
                std::printf("rescan         %d FOVs re-focused, %d now pass (%d still flagged)\n",
                            rescanned, rescan_fixed, rescanned - rescan_fixed);
                for (size_t ti : flagged)
                    std::printf("               FOV (%d,%d): %d/%d tissue patches blurred, "
                                "focus error after rescan %.2f um\n",
                                acq[ti].x, acq[ti].y, blurred_in_tile[ti], tissue_in_tile[ti],
                                acq[ti].af_error_um);
            } else {
                std::printf("rescan         nothing flagged as blurred\n");
            }
        }
    }

    // ---- 10. artefacts on disk -------------------------------------------
    if (args.save_preview) {
        write_pnm(args.out + "/thumb.ppm", downsample(slide, args.thumb_scale));
        write_pnm(args.out + "/mask.pgm", mask_to_image(seg.mask));
        write_pnm(args.out + "/mosaic.ppm", downsample(canvas, 8));
    }

    const double total_ms = ms_since(t_start);
    std::string json = args.out + "/report.json";
    std::ofstream j(json);
    j.setf(std::ios::fixed);
    j.precision(4);
    j << "{\n";
    j << "  \"slide\": {\"path\": \"" << args.slide << "\", \"width\": " << slide.w
      << ", \"height\": " << slide.h << "},\n";
    j << "  \"segmentation\": {\"otsu_threshold\": " << seg.threshold
      << ", \"tissue_fraction\": " << seg.tissue_frac << ", \"ms\": " << seg_ms << "},\n";
    j << "  \"plan\": {\"cols\": " << plan.cols << ", \"rows\": " << plan.rows
      << ", \"total_fovs\": " << plan.total << ", \"acquired\": " << plan.to_acquire
      << ", \"skipped_fraction\": " << skipped_frac << ", \"fov\": " << args.fov
      << ", \"overlap\": " << args.overlap << "},\n";
    j << "  \"flatfield\": {\"corner_centre_before\": " << ff_before
      << ", \"corner_centre_after\": " << ff_after << "},\n";
    j << "  \"autofocus\": {\"metric\": \"" << metric_name(args.metric) << "\", \"map_enabled\": "
      << (args.use_map ? "true" : "false") << ", \"map_mode\": \""
      << (args.map_mode == FocusMap::Mode::Plane ? "plane" : "plane+local") << "\""
      << ", \"verify_span_um\": " << args.verify_span << ", \"tiles\": " << acq.size()
      << ", \"full_sweeps\": " << full_sweeps << ", \"from_map\": " << mapped
      << ", \"pct_full_sweep\": " << pct_full << ", \"frames\": " << af_frames
      << ", \"frames_per_tile\": " << (acq.empty() ? 0.0 : double(af_frames) / double(acq.size()))
      << ", \"search_ms\": " << af_time_ms
      << ", \"search_ms_per_tile\": " << (acq.empty() ? 0.0 : af_time_ms / double(acq.size()))
      << ", \"focus_error_um_mean\": " << mean(af_errors)
      << ", \"focus_error_um_p95\": " << percentile(af_errors, 0.95)
      << ", \"focus_error_um_mean_mapped\": " << mean(af_errors_mapped)
      << ", \"focus_error_um_mean_swept\": " << mean(af_errors_swept)
      << ", \"map_points\": " << map.size() << ", \"map_residual_um\": " << map.rms_residual()
      << "},\n";
    j << "  \"instrument_time_ms\": {\"stage\": " << stage.stats().move_time_ms
      << ", \"exposure\": " << cam.stats().exposure_time_ms
      << ", \"total\": " << stage.stats().move_time_ms + cam.stats().exposure_time_ms
      << ", \"exposures\": " << cam.stats().exposures << "},\n";
    j << "  \"registration\": {\"patch\": " << patch << ", \"pairs\": " << pairs
      << ", \"error_px_mean\": " << mean(reg_err)
      << ", \"error_px_median\": " << percentile(reg_err, 0.5)
      << ", \"error_px_p95\": " << percentile(reg_err, 0.95)
      << ", \"pairs_over_1px\": " << pair_failures << ", \"weak_links_rejected\": " << weak_links
      << ", \"peak_median\": " << percentile(reg_peak, 0.5)
      << ", \"peak_p5\": " << percentile(reg_peak, 0.05) << ", \"ms\": " << reg_ms
      << ", \"ms_per_pair\": " << (pairs ? reg_ms / pairs : 0.0) << "},\n";
    j << "  \"mosaic\": {\"width\": " << canvas_w << ", \"height\": " << canvas_h
      << ", \"registration_components\": " << components
      << ", \"placement_error_px_mean\": " << mean(place_err)
      << ", \"placement_error_px_max\": "
      << (place_err.empty() ? 0.0 : *std::max_element(place_err.begin(), place_err.end()))
      << ", \"ms\": " << mosaic_ms << "},\n";
    j << "  \"pyramid\": {\"levels\": " << pyr.levels.size() << ", \"tile\": " << pyr.tile_size
      << ", \"bytes\": " << pyr.bytes_written << ", \"resample_ms\": " << pyr.resample_ms
      << ", \"resample_gbs\": " << pyr.resample_gbs << ", \"write_ms\": " << pyr.write_ms
      << ", \"dims\": [";
    for (size_t i = 0; i < pyr.levels.size(); ++i)
        j << (i ? ", " : "") << "[" << pyr.levels[i].w << ", " << pyr.levels[i].h << "]";
    j << "]},\n";
    j << "  \"quality\": {\"status\": \"" << qc_status << "\", \"patches\": " << patch_count
      << ", \"tiles_per_s\": " << qc_tiles_per_s << ", \"infer_ms\": " << qc_ms
      << ", \"threads\": " << args.threads << ", \"flagged_fovs\": " << flagged_fovs
      << ", \"counts\": {";
    {
        bool first = true;
        for (const auto& [name, n] : class_counts) {
            j << (first ? "" : ", ") << "\"" << name << "\": " << n;
            first = false;
        }
    }
    j << "}, \"fold_detected\": " << fold_hits << ", \"fold_truth\": " << fold_truth
      << ", \"bubble_detected\": " << bubble_hits << ", \"bubble_truth\": " << bubble_truth
      << ", \"rescanned\": " << rescanned << ", \"rescan_fixed\": " << rescan_fixed << "},\n";
    j << "  \"wall_ms\": {\"total\": " << total_ms << ", \"acquire\": " << acquire_ms
      << ", \"registration\": " << reg_ms << ", \"mosaic\": " << mosaic_ms << "}\n";
    j << "}\n";
    j.close();

    std::printf("done           %.1f s wall -> %s, %s\n", total_ms / 1000.0, tif_path.c_str(),
                json.c_str());
    return 0;
}
