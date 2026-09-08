#include "wsi/classifier.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>

#ifdef WSI_WITH_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace wsi {

Image prepare_tile(const Image& src, int size) {
    Image gray = to_gray(src);
    // Centre square crop first, so the aspect ratio is never distorted.
    int side = std::min(gray.w, gray.h);
    Image sq = crop(gray, (gray.w - side) / 2, (gray.h - side) / 2, side, side);
    Image out(size, size, 1);
    // Box resample. Nearest would alias exactly the high-frequency detail the
    // model is being asked to judge.
    for (int y = 0; y < size; ++y) {
        int sy0 = y * side / size, sy1 = std::max(sy0 + 1, (y + 1) * side / size);
        for (int x = 0; x < size; ++x) {
            int sx0 = x * side / size, sx1 = std::max(sx0 + 1, (x + 1) * side / size);
            unsigned acc = 0, n = 0;
            for (int sy = sy0; sy < sy1; ++sy)
                for (int sx = sx0; sx < sx1; ++sx) {
                    acc += sq.at(sx, sy);
                    ++n;
                }
            out.at(x, y) = uint8_t(acc / std::max(1u, n));
        }
    }
    return out;
}

#ifdef WSI_WITH_ONNX

namespace {

class OnnxTileClassifier final : public ITileClassifier {
public:
    OnnxTileClassifier(const std::string& model_path, int threads)
        : env_(ORT_LOGGING_LEVEL_WARNING, "wsi") {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        std::wstring wpath(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), opts);
#else
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
#endif
        Ort::AllocatorWithDefaultOptions alloc;
        in_name_ = session_->GetInputNameAllocated(0, alloc).get();
        out_name_ = session_->GetOutputNameAllocated(0, alloc).get();

        auto shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        // NCHW with a dynamic batch: the trailing dimension is the tile size.
        input_size_ = shape.size() == 4 && shape[3] > 0 ? int(shape[3]) : 64;

        auto oshape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        int n_classes = oshape.size() == 2 && oshape[1] > 0 ? int(oshape[1]) : 5;
        // Must match CLASSES in python/train_quality.py, in order.
        static const char* kNames[] = {"sharp", "blurred", "folded", "bubble", "background"};
        const int kKnown = int(sizeof(kNames) / sizeof(kNames[0]));
        for (int i = 0; i < n_classes; ++i)
            names_.push_back(i < kKnown ? kNames[i] : "class" + std::to_string(i));
    }

    std::vector<TileQuality> classify(const std::vector<Image>& tiles) override {
        std::vector<TileQuality> out(tiles.size());
        if (tiles.empty()) return out;

        const int s = input_size_;
        std::vector<float> in(tiles.size() * size_t(s) * s);
        for (size_t i = 0; i < tiles.size(); ++i) {
            Image t = prepare_tile(tiles[i], s);
            for (int p = 0; p < s * s; ++p)
                in[i * size_t(s) * s + p] = float(t.px[size_t(p)]) / 255.0f;
        }

        std::array<int64_t, 4> shape{int64_t(tiles.size()), 1, s, s};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(), shape.data(),
                                                           shape.size());
        const char* in_names[] = {in_name_.c_str()};
        const char* out_names[] = {out_name_.c_str()};

        auto t0 = std::chrono::steady_clock::now();
        auto res = session_->Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
        auto t1 = std::chrono::steady_clock::now();
        st_.infer_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        st_.tiles += tiles.size();

        const float* logits = res[0].GetTensorData<float>();
        const size_t k = names_.size();
        for (size_t i = 0; i < tiles.size(); ++i) {
            const float* row = logits + i * k;
            size_t best = 0;
            for (size_t j = 1; j < k; ++j)
                if (row[j] > row[best]) best = j;
            double sum = 0;
            for (size_t j = 0; j < k; ++j) sum += std::exp(double(row[j]) - double(row[best]));
            out[i].cls = int(best);
            out[i].prob = float(1.0 / sum);
        }
        return out;
    }

    const std::vector<std::string>& class_names() const override { return names_; }
    ClassifierStats stats() const override { return st_; }
    int input_size() const override { return input_size_; }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::string in_name_, out_name_;
    std::vector<std::string> names_;
    int input_size_ = 64;
    ClassifierStats st_;
};

}  // namespace

std::unique_ptr<ITileClassifier> make_onnx_classifier(const std::string& model_path, int threads,
                                                      std::string* error) {
    try {
        return std::make_unique<OnnxTileClassifier>(model_path, threads);
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return nullptr;
    }
}

#else

std::unique_ptr<ITileClassifier> make_onnx_classifier(const std::string&, int, std::string* error) {
    if (error) *error = "built without ONNX Runtime (configure with -DWSI_WITH_ONNX=ON)";
    return nullptr;
}

#endif

}  // namespace wsi
