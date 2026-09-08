// Tile-quality classifier: in-focus / blurred / folded / bubble.
//
// The model is trained in PyTorch (python/train_quality.py) on tiles degraded
// synthetically, exported to ONNX, and run here through ONNX Runtime so the
// scanner can decide to rescan a region while the slide is still on the stage.
// Built only when WSI_WITH_ONNX is on; otherwise the factory returns nullptr
// and the scanner runs without the quality-control pass.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "wsi/image.hpp"

namespace wsi {

struct TileQuality {
    int cls = 0;
    float prob = 0.f;
};

struct ClassifierStats {
    uint64_t tiles = 0;
    double infer_ms = 0;
};

class ITileClassifier {
public:
    virtual ~ITileClassifier() = default;
    // Each input is resized to the model's input size internally.
    virtual std::vector<TileQuality> classify(const std::vector<Image>& tiles) = 0;
    virtual const std::vector<std::string>& class_names() const = 0;
    virtual ClassifierStats stats() const = 0;
    virtual int input_size() const = 0;
};

// nullptr when the build has no ONNX Runtime, or the model failed to load.
std::unique_ptr<ITileClassifier> make_onnx_classifier(const std::string& model_path,
                                                      int intra_op_threads = 1,
                                                      std::string* error = nullptr);

// Centre crop + box resize to a square gray tile, which is what the model eats.
Image prepare_tile(const Image& src, int size);

}  // namespace wsi
