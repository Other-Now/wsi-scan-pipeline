// splitmix64, used two ways: as a cheap PRNG, and as a hash so that a "random"
// property of a site (stage jitter, per-FOV illumination gain) is the same
// every time that site is visited. Autofocus revisits sites; if the jitter
// changed per grab there would be no ground truth to score stitching against.
#pragma once
#include <cmath>
#include <cstdint>

namespace wsi {

inline uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline uint64_t hash_site(uint64_t seed, int64_t a, int64_t b) {
    uint64_t s = seed ^ (uint64_t(a) * 0x9E3779B97F4A7C15ULL) ^ (uint64_t(b) * 0xC2B2AE3D27D4EB4FULL);
    return splitmix64(s);
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 1) {}
    uint64_t next() { return splitmix64(s); }
    double uniform() { return double(next() >> 11) * (1.0 / 9007199254740992.0); }
    double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }
    // Box-Muller, one value per call; good enough for read noise.
    double normal() {
        double u1 = uniform(), u2 = uniform();
        if (u1 < 1e-12) u1 = 1e-12;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
    int uniform_int(int lo, int hi) { return lo + int(next() % uint64_t(hi - lo + 1)); }
};

}  // namespace wsi
