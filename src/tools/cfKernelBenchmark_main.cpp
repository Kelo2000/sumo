/****************************************************************************/
// Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.dev/sumo
// Copyright (C) 2001-2025 German Aerospace Center (DLR) and others.
// This program and the accompanying materials are made available under the
// terms of the Eclipse Public License 2.0 which is available at
// https://www.eclipse.org/legal/epl-2.0/
// This Source Code may also be made available under the following Secondary
// Licenses when the conditions for such availability set forth in the Eclipse
// Public License 2.0 are satisfied: GNU General Public License, version 2
// or later which is available at
// https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
/****************************************************************************/
/// @file    cfKernelBenchmark_main.cpp
/// @author  Eclipse SUMO contributors
/// @date    2025-01-01
/// @brief   Micro-benchmark for the header-only CF kernel (scalar vs SIMD).
/****************************************************************************/
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <immintrin.h>
#include <random>
#include <string>
#include <vector>

#include <microsim/cfmodels/MSCFKernel.h>
#include <microsim/cfmodels/MSCFModel_IDM.h>
#include <microsim/MSGlobals.h>
#include <microsim/MSVehicleType.h>
#include <utils/common/SUMOTime.h>
#include <utils/vehicle/SUMOVTypeParameter.h>
#include <utils/xml/SUMOXMLDefinitions.h>

namespace {

struct BenchmarkConfig {
    std::size_t batchSize = 1 << 20;
    int iterations = 5;
    bool semiImplicitEuler = true;
};

BenchmarkConfig parseArgs(int argc, char** argv) {
    BenchmarkConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--ballistic") {
            cfg.semiImplicitEuler = false;
        } else if (arg == "--euler") {
            cfg.semiImplicitEuler = true;
        } else if (arg == "--batch" && i + 1 < argc) {
            cfg.batchSize = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--iters" && i + 1 < argc) {
            cfg.iterations = std::stoi(argv[++i]);
        }
    }
    return cfg;
}

double runScalar(const sumo::cfkernel::Limits& limits,
                 const std::vector<double>& gaps,
                 const std::vector<double>& speeds,
                 const std::vector<double>& leaderSpeeds,
                 const std::vector<double>& leaderDecels,
                 std::vector<double>& out) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < gaps.size(); ++i) {
        out[i] = sumo::cfkernel::maximumSafeFollowSpeed(
            gaps[i], speeds[i], leaderSpeeds[i], leaderDecels[i], false, limits);
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

double runModel(const MSCFModel& model,
                const std::vector<double>& gaps,
                const std::vector<double>& speeds,
                const std::vector<double>& leaderSpeeds,
                const std::vector<double>& leaderDecels,
                std::vector<double>& out) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < gaps.size(); ++i) {
        out[i] = model.maximumSafeFollowSpeed(gaps[i], speeds[i], leaderSpeeds[i], leaderDecels[i], false);
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

inline double maxAbsDiff(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    double maxDiff = 0.;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        maxDiff = std::max(maxDiff, std::abs(lhs[i] - rhs[i]));
    }
    return maxDiff;
}

#ifdef __AVX__
inline __m256d mm256_max_pd(__m256d a, __m256d b) {
    return _mm256_max_pd(a, b);
}

inline __m256d mm256_min_pd(__m256d a, __m256d b) {
    return _mm256_min_pd(a, b);
}

inline __m256d mm256_blendv_pd(__m256d a, __m256d b, __m256d mask) {
    return _mm256_blendv_pd(a, b, mask);
}

inline __m256d mm256_cmpgt_pd(__m256d a, __m256d b) {
    return _mm256_cmp_pd(a, b, _CMP_GT_OQ);
}

inline __m256d mm256_cmplt_pd(__m256d a, __m256d b) {
    return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
}

inline __m256d mm256_fmadd_pd(__m256d a, __m256d b, __m256d c) {
#ifdef __FMA__
    return _mm256_fmadd_pd(a, b, c);
#else
    return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#endif
}

inline __m256d brakeGapEulerVec(__m256d speed, __m256d decel, __m256d headwayTime, __m256d dt) {
    const __m256d speedReduction = _mm256_mul_pd(decel, dt);
    const __m256d steps = _mm256_div_pd(speed, speedReduction);
    const __m256i stepsInt = _mm256_cvttpd_epi32(steps);
    const __m256d stepsD = _mm256_cvtepi32_pd(stepsInt);
    const __m256d stepsPlus = _mm256_add_pd(stepsD, _mm256_set1_pd(1.0));
    const __m256d term = _mm256_sub_pd(_mm256_mul_pd(stepsD, speed),
                                       _mm256_mul_pd(speedReduction,
                                                     _mm256_mul_pd(stepsD, _mm256_mul_pd(stepsPlus, _mm256_set1_pd(0.5)))));
    const __m256d dist = _mm256_mul_pd(term, dt);
    return _mm256_add_pd(dist, _mm256_mul_pd(speed, headwayTime));
}

inline __m256d maximumSafeStopSpeedEulerVec(__m256d gap, __m256d decel, __m256d headwayTime, __m256d dt) {
    const __m256d eps = _mm256_set1_pd(NUMERICAL_EPS);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d g = _mm256_sub_pd(gap, eps);
    const __m256d gMask = mm256_cmplt_pd(g, zero);
    const __m256d b = _mm256_mul_pd(decel, dt);
    const __m256d t = headwayTime;
    const __m256d s = dt;
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d four = _mm256_set1_pd(4.0);
    const __m256d half = _mm256_set1_pd(0.5);

    const __m256d twoGOverB = _mm256_div_pd(_mm256_mul_pd(two, g), b);
    const __m256d inner = _mm256_add_pd(_mm256_mul_pd(s, _mm256_sub_pd(twoGOverB, t)),
                                        _mm256_mul_pd(t, t));
    const __m256d sqrtTerm = _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(s, s), _mm256_mul_pd(four, inner)));
    const __m256d nRaw = _mm256_sub_pd(half,
                                       _mm256_div_pd(_mm256_add_pd(t, _mm256_mul_pd(sqrtTerm, _mm256_set1_pd(-0.5))), s));
    const __m256d n = _mm256_floor_pd(nRaw);
    const __m256d nMinus = _mm256_sub_pd(n, _mm256_set1_pd(1.0));
    const __m256d h = mm256_fmadd_pd(_mm256_mul_pd(half, _mm256_mul_pd(n, nMinus)), _mm256_mul_pd(b, s),
                                     _mm256_mul_pd(n, _mm256_mul_pd(b, t)));
    const __m256d r = _mm256_div_pd(_mm256_sub_pd(g, h), _mm256_add_pd(_mm256_mul_pd(n, s), t));
    const __m256d x = _mm256_add_pd(_mm256_mul_pd(n, b), r);
    return mm256_blendv_pd(x, zero, gMask);
}

double runSimdIntrinsicEuler(const sumo::cfkernel::Limits& limits,
                             const std::vector<double>& gaps,
                             const std::vector<double>& speeds,
                             const std::vector<double>& leaderSpeeds,
                             const std::vector<double>& leaderDecels,
                             std::vector<double>& out) {
    const auto start = std::chrono::steady_clock::now();
    const std::size_t size = gaps.size();
    const __m256d decel = _mm256_set1_pd(limits.decel);
    const __m256d headway = _mm256_set1_pd(limits.headwayTime);
    const __m256d dt = _mm256_set1_pd(limits.timeStep);
    const __m256d eps = _mm256_set1_pd(NUMERICAL_EPS);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d amp = _mm256_set1_pd(sumo::cfkernel::kEmergencyDecelAmplifier);

    std::size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d gap = _mm256_loadu_pd(&gaps[i]);
        __m256d egoSpeed = _mm256_loadu_pd(&speeds[i]);
        __m256d predSpeed = _mm256_loadu_pd(&leaderSpeeds[i]);
        __m256d predDecel = _mm256_loadu_pd(&leaderDecels[i]);

        const __m256d maxDecel = mm256_max_pd(decel, predDecel);
        const __m256d brakeGap = brakeGapEulerVec(predSpeed, maxDecel, zero, dt);
        const __m256d gapStop = _mm256_add_pd(gap, brakeGap);
        __m256d x = maximumSafeStopSpeedEulerVec(gapStop, decel, headway, dt);

        const __m256d origSafeDecel = _mm256_div_pd(_mm256_sub_pd(egoSpeed, x), dt);
        const __m256d decelLimit = _mm256_add_pd(decel, eps);
        const __m256d needEmergencyMask = mm256_cmpgt_pd(origSafeDecel, decelLimit);

        const __m256d predBrakeDist = _mm256_div_pd(_mm256_mul_pd(_mm256_mul_pd(predSpeed, predSpeed), _mm256_set1_pd(0.5)),
                                                    predDecel);
        const __m256d b1 = _mm256_div_pd(_mm256_mul_pd(_mm256_mul_pd(egoSpeed, egoSpeed), _mm256_set1_pd(0.5)),
                                         _mm256_add_pd(gap, predBrakeDist));
        __m256d safeDecel = mm256_max_pd(b1, predDecel);
        safeDecel = _mm256_mul_pd(safeDecel, amp);
        safeDecel = mm256_max_pd(safeDecel, decel);
        safeDecel = mm256_min_pd(safeDecel, origSafeDecel);
        __m256d xEmergency = _mm256_sub_pd(egoSpeed, _mm256_mul_pd(safeDecel, dt));
        xEmergency = mm256_max_pd(xEmergency, zero);
        x = mm256_blendv_pd(x, xEmergency, needEmergencyMask);

        _mm256_storeu_pd(&out[i], x);
    }
    for (; i < size; ++i) {
        out[i] = sumo::cfkernel::maximumSafeFollowSpeed(
            gaps[i], speeds[i], leaderSpeeds[i], leaderDecels[i], false, limits);
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}
#endif

double runSimd(const sumo::cfkernel::Limits& limits,
               const std::vector<double>& gaps,
               const std::vector<double>& speeds,
               const std::vector<double>& leaderSpeeds,
               const std::vector<double>& leaderDecels,
               std::vector<double>& out) {
    const auto start = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::size_t i = 0; i < gaps.size(); ++i) {
        out[i] = sumo::cfkernel::maximumSafeFollowSpeed(
            gaps[i], speeds[i], leaderSpeeds[i], leaderDecels[i], false, limits);
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

} // namespace

int main(int argc, char** argv) {
    const BenchmarkConfig cfg = parseArgs(argc, argv);
    DELTA_T = 1000;
    MSGlobals::gActionStepLength = DELTA_T;
    MSGlobals::gSemiImplicitEulerUpdate = cfg.semiImplicitEuler;
    MSGlobals::gComputeLC = false;
    SUMOVTypeParameter params("bench", SVC_PASSENGER);
    params.cfParameter[SUMO_ATTR_DECEL] = "2.6";
    params.cfParameter[SUMO_ATTR_EMERGENCYDECEL] = "7.5";
    params.cfParameter[SUMO_ATTR_TAU] = "1.0";
    MSVehicleType vtype(params);
    MSCFModel_IDM model(&vtype, false);
    const sumo::cfkernel::Limits limits{
        2.6,   // decel
        7.5,   // emergencyDecel
        1.0,   // headwayTime
        TS,    // timeStep
        cfg.semiImplicitEuler,
        false  // computeLC
    };

    std::vector<double> gaps(cfg.batchSize);
    std::vector<double> speeds(cfg.batchSize);
    std::vector<double> leaderSpeeds(cfg.batchSize);
    std::vector<double> leaderDecels(cfg.batchSize);
    std::vector<double> outScalar(cfg.batchSize);
    std::vector<double> outSimd(cfg.batchSize);
    std::vector<double> outModel(cfg.batchSize);
    std::vector<double> outIntrinsic(cfg.batchSize);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> speedDist(0.0, 35.0);
    std::uniform_real_distribution<double> gapDist(0.0, 120.0);
    std::uniform_real_distribution<double> leaderDecelDist(1.0, 6.0);

    for (std::size_t i = 0; i < cfg.batchSize; ++i) {
        speeds[i] = speedDist(rng);
        leaderSpeeds[i] = speedDist(rng);
        gaps[i] = gapDist(rng);
        leaderDecels[i] = leaderDecelDist(rng);
    }

    double scalarTime = 0.;
    double simdTime = 0.;
    double modelTime = 0.;
    double intrinsicTime = 0.;
    for (int iter = 0; iter < cfg.iterations; ++iter) {
        scalarTime += runScalar(limits, gaps, speeds, leaderSpeeds, leaderDecels, outScalar);
        modelTime += runModel(model, gaps, speeds, leaderSpeeds, leaderDecels, outModel);
        simdTime += runSimd(limits, gaps, speeds, leaderSpeeds, leaderDecels, outSimd);
#ifdef __AVX__
        if (cfg.semiImplicitEuler) {
            intrinsicTime += runSimdIntrinsicEuler(limits, gaps, speeds, leaderSpeeds, leaderDecels, outIntrinsic);
        }
#endif
    }

    const double scalarAvg = scalarTime / cfg.iterations;
    const double simdAvg = simdTime / cfg.iterations;
    const double modelAvg = modelTime / cfg.iterations;
    const double intrinsicAvg = intrinsicTime / cfg.iterations;
    const double scalarThroughput = cfg.batchSize / scalarAvg / 1e6;
    const double simdThroughput = cfg.batchSize / simdAvg / 1e6;
    const double modelThroughput = cfg.batchSize / modelAvg / 1e6;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CF kernel benchmark (" << (cfg.semiImplicitEuler ? "Euler" : "Ballistic")
              << ", batch=" << cfg.batchSize << ", iters=" << cfg.iterations << ")\n";
    std::cout << "Scalar avg: " << scalarAvg << " s, throughput: " << scalarThroughput << " Mveh/s\n";
    std::cout << "IDM model avg: " << modelAvg << " s, throughput: " << modelThroughput << " Mveh/s\n";
    std::cout << "SIMD   avg: " << simdAvg << " s, throughput: " << simdThroughput << " Mveh/s\n";
    std::cout << "Max abs diff (kernel scalar vs IDM): " << maxAbsDiff(outScalar, outModel) << "\n";
    std::cout << "Max abs diff (kernel SIMD vs IDM): " << maxAbsDiff(outSimd, outModel) << "\n";
#ifdef __AVX__
    const double intrinsicThroughput = intrinsicAvg > 0. ? cfg.batchSize / intrinsicAvg / 1e6 : 0.;
    if (cfg.semiImplicitEuler) {
        std::cout << "Intrinsic avg: " << intrinsicAvg << " s, throughput: " << intrinsicThroughput << " Mveh/s\n";
        std::cout << "Max abs diff (intrinsic vs IDM): " << maxAbsDiff(outIntrinsic, outModel) << "\n";
    } else {
        std::cout << "Intrinsic avg: n/a (ballistic)\n";
    }
#else
    std::cout << "Intrinsic avg: n/a (no AVX)\n";
#endif
    return 0;
}
