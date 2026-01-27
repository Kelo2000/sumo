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
#include <random>
#include <string>
#include <vector>

#include <microsim/cfmodels/MSCFKernel.h>

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
    const sumo::cfkernel::Limits limits{
        2.6,   // decel
        7.5,   // emergencyDecel
        1.0,   // headwayTime
        1.0,   // timeStep
        cfg.semiImplicitEuler,
        false  // computeLC
    };

    std::vector<double> gaps(cfg.batchSize);
    std::vector<double> speeds(cfg.batchSize);
    std::vector<double> leaderSpeeds(cfg.batchSize);
    std::vector<double> leaderDecels(cfg.batchSize);
    std::vector<double> outScalar(cfg.batchSize);
    std::vector<double> outSimd(cfg.batchSize);

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
    for (int iter = 0; iter < cfg.iterations; ++iter) {
        scalarTime += runScalar(limits, gaps, speeds, leaderSpeeds, leaderDecels, outScalar);
        simdTime += runSimd(limits, gaps, speeds, leaderSpeeds, leaderDecels, outSimd);
    }

    double maxAbsDiff = 0.;
    for (std::size_t i = 0; i < cfg.batchSize; ++i) {
        maxAbsDiff = std::max(maxAbsDiff, std::abs(outScalar[i] - outSimd[i]));
    }

    const double scalarAvg = scalarTime / cfg.iterations;
    const double simdAvg = simdTime / cfg.iterations;
    const double scalarThroughput = cfg.batchSize / scalarAvg / 1e6;
    const double simdThroughput = cfg.batchSize / simdAvg / 1e6;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CF kernel benchmark (" << (cfg.semiImplicitEuler ? "Euler" : "Ballistic")
              << ", batch=" << cfg.batchSize << ", iters=" << cfg.iterations << ")\n";
    std::cout << "Scalar avg: " << scalarAvg << " s, throughput: " << scalarThroughput << " Mveh/s\n";
    std::cout << "SIMD   avg: " << simdAvg << " s, throughput: " << simdThroughput << " Mveh/s\n";
    std::cout << "Max abs diff (scalar vs SIMD): " << maxAbsDiff << "\n";
    return 0;
}
