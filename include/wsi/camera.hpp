// Camera and stage abstractions, plus the synthetic backend the whole project
// runs on.
//
// The scanner talks only to ICamera / IStage. Swapping in a real camera SDK
// (or a robot stage) is a matter of writing one more implementation of these
// two interfaces -- nothing in the scan loop, autofocus, or stitching knows
// where the pixels came from.
#pragma once
#include <cstdint>
#include <memory>
#include <utility>

#include "wsi/image.hpp"

namespace wsi {

struct StageStats {
    uint64_t xy_moves = 0;
    uint64_t z_moves = 0;
    double travel_px = 0;
    double travel_z_um = 0;
    double move_time_ms = 0;  // simulated instrument time, from a cost model
};

class IStage {
public:
    virtual ~IStage() = default;
    virtual void move_xy(int x, int y) = 0;  // top-left of the FOV, in slide pixels
    virtual void move_z(double z_um) = 0;
    virtual int x() const = 0;
    virtual int y() const = 0;
    virtual double z() const = 0;
    virtual StageStats stats() const = 0;
};

struct CameraStats {
    uint64_t exposures = 0;
    uint64_t pixels = 0;
    double exposure_time_ms = 0;  // simulated instrument time
};

class ICamera {
public:
    virtual ~ICamera() = default;
    virtual Image grab() = 0;                  // full FOV at the current stage pose
    virtual Image grab_roi(int w, int h) = 0;  // centred autofocus ROI, same optics
    virtual int fov_w() const = 0;
    virtual int fov_h() const = 0;
    virtual CameraStats stats() const = 0;
};

class SyntheticStage final : public IStage {
public:
    // Cost model: the stage covers `speed_px_per_ms` and then needs `settle_ms`
    // to stop ringing. Z is slower per unit but travels far less.
    SyntheticStage(double speed_px_per_ms = 8.0, double settle_ms = 4.0,
                   double z_ms_per_um = 0.35, double z_settle_ms = 2.0);

    void move_xy(int x, int y) override;
    void move_z(double z_um) override;
    int x() const override { return x_; }
    int y() const override { return y_; }
    double z() const override { return z_; }
    StageStats stats() const override { return st_; }

private:
    int x_ = 0, y_ = 0;
    double z_ = 0;
    double speed_, settle_, z_rate_, z_settle_;
    StageStats st_;
};

struct OpticsParams {
    int fov_w = 512, fov_h = 512;

    // The focus surface: a tilted plane plus a slow bow. A plane fit tracks it
    // almost everywhere, which is what makes a focus map worth building.
    double z_tilt_x = 0.0022;   // um of best focus per slide pixel in x
    double z_tilt_y = -0.0016;
    double z_offset = 0.0;
    double z_bow = 1.2;         // um, amplitude of the sinusoidal warp
    double z_bow_period = 2600; // slide pixels

    // A local anomaly (debris under the coverslip). The plane fit cannot see it,
    // so tiles here come out blurred -- that is what the tile-quality classifier
    // is for, and what triggers a rescan.
    int anomaly_x = -1, anomaly_y = -1, anomaly_r = 0;
    double anomaly_dz = 0.0;    // um

    double blur_px_per_um = 0.55;  // defocus blur sigma per um off best focus
    double blur_cap = 8.0;
    double vignette = 0.35;        // corner falloff fraction
    double illum_gain_jitter = 0.05;
    double noise_sigma = 2.0;      // read noise, 8-bit counts
    double xy_jitter_px = 3.0;     // stage positioning error, per site
    double exposure_ms = 12.0;
    uint64_t seed = 7;
};

class SyntheticCamera final : public ICamera {
public:
    SyntheticCamera(const Image& slide, const IStage& stage, OpticsParams p);

    Image grab() override;
    Image grab_roi(int w, int h) override;
    int fov_w() const override { return p_.fov_w; }
    int fov_h() const override { return p_.fov_h; }
    CameraStats stats() const override { return st_; }

    // A white-light calibration frame with no slide in the path.
    Image blank_frame() const;

    // Oracle access. Used only to score the run, never by the scan loop.
    double true_focus_z(int x, int y) const;
    std::pair<int, int> true_offset(int x, int y) const;

    const OpticsParams& params() const { return p_; }

private:
    Image render(int out_w, int out_h, int centre_bias_x, int centre_bias_y);

    const Image& slide_;
    const IStage& stage_;
    OpticsParams p_;
    CameraStats st_;
    uint64_t noise_state_;
};

}  // namespace wsi
