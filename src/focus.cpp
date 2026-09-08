#include "wsi/focus.hpp"

namespace wsi {

double brenner(const Image& g, int lag) {
    double acc = 0.0;
    long long n = 0;
    for (int y = 0; y < g.h; ++y) {
        for (int x = 0; x + lag < g.w; ++x) {
            double d = double(g.at(x + lag, y)) - double(g.at(x, y));
            acc += d * d;
            ++n;
        }
    }
    return n ? acc / double(n) : 0.0;
}

double var_laplacian(const Image& g) {
    double sum = 0.0, sum2 = 0.0;
    long long n = 0;
    for (int y = 1; y < g.h - 1; ++y) {
        for (int x = 1; x < g.w - 1; ++x) {
            double l = 4.0 * g.at(x, y) - g.at(x - 1, y) - g.at(x + 1, y) - g.at(x, y - 1) -
                       g.at(x, y + 1);
            sum += l;
            sum2 += l * l;
            ++n;
        }
    }
    if (!n) return 0.0;
    double mean = sum / double(n);
    return sum2 / double(n) - mean * mean;
}

double focus_score(const Image& g, FocusMetric m) {
    return m == FocusMetric::Brenner ? brenner(g) : var_laplacian(g);
}

const char* metric_name(FocusMetric m) {
    return m == FocusMetric::Brenner ? "brenner" : "var_laplacian";
}

}  // namespace wsi
