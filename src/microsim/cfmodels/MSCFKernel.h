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
/// @file    MSCFKernel.h
/// @author  Eclipse SUMO contributors
/// @date    2025-01-01
/// @brief   Header-only, kernel-ready car-following helpers mirroring MSCFModel.
/****************************************************************************/
#pragma once

#include <cmath>
#include <config.h>
#include <algorithm>

namespace sumo::cfkernel {

struct Limits {
    double decel;
    double emergencyDecel;
    double headwayTime;
    double timeStep;
    bool semiImplicitEuler;
    bool computeLC;
};

inline double accelToSpeed(const double accel, const double dt) {
    return accel * dt;
}

inline double speedToAccel(const double speed, const double dt) {
    return speed / dt;
}

inline double speedToDist(const double speed, const double dt) {
    return speed * dt;
}

inline double distToSpeed(const double dist, const double dt) {
    return dist / dt;
}

inline double brakeGapEuler(const double speed, const double decel, const double headwayTime, const double dt) {
    const double speedReduction = accelToSpeed(decel, dt);
    const int steps = static_cast<int>(speed / speedReduction);
    return speedToDist(steps * speed - speedReduction * steps * (steps + 1) / 2, dt) + speed * headwayTime;
}

inline double brakeGap(const double speed, const double decel, const double headwayTime, const Limits& limits) {
    if (limits.semiImplicitEuler) {
        return brakeGapEuler(speed, decel, headwayTime, limits.timeStep);
    }
    if (speed <= 0.) {
        return 0.;
    }
    return speed * (headwayTime + 0.5 * speed / decel);
}

inline double maximumSafeStopSpeedEuler(double gap, double decel, double headway, const Limits& limits) {
    const double g = gap - NUMERICAL_EPS;
    if (g < 0.) {
        return 0.;
    }
    const double b = accelToSpeed(decel, limits.timeStep);
    const double t = headway >= 0 ? headway : limits.headwayTime;
    const double s = limits.timeStep;
    const double n = std::floor(.5 - ((t + (std::sqrt(((s * s) + (4.0 * ((s * (2.0 * g / b - t)) + (t * t))))) * -0.5)) / s));
    const double h = 0.5 * n * (n - 1) * b * s + n * b * t;
    const double r = (g - h) / (n * s + t);
    const double x = n * b + r;
    return x;
}

inline double maximumSafeStopSpeedBallistic(double gap, double decel, double currentSpeed, bool onInsertion, double headway, const Limits& limits) {
    const double g = std::max(0., gap - NUMERICAL_EPS);
    headway = headway >= 0 ? headway : limits.headwayTime;

    if (onInsertion) {
        const double btau = decel * headway;
        const double v0 = -btau + std::sqrt(btau * btau + 2 * decel * g);
        return v0;
    }

    const double tau = headway == 0 ? limits.timeStep : headway;
    const double v0 = std::max(0., currentSpeed);
    if (g <= v0 * tau * 0.5) {
        if (g == 0.) {
            if (v0 > 0.) {
                return -accelToSpeed(limits.emergencyDecel, limits.timeStep);
            }
            return 0.;
        }
        const double a = -v0 * v0 / (2 * g);
        return v0 + a * limits.timeStep;
    }

    const double btau2 = decel * tau / 2;
    const double v1 = -btau2 + std::sqrt(btau2 * btau2 + decel * (2 * g - tau * v0));
    const double a = (v1 - v0) / tau;
    return v0 + a * limits.timeStep;
}

inline double maximumSafeStopSpeed(double gap, double decel, double currentSpeed, bool onInsertion, double headway, const Limits& limits) {
    if (limits.semiImplicitEuler) {
        return maximumSafeStopSpeedEuler(gap, decel, headway, limits);
    }
    return maximumSafeStopSpeedBallistic(gap, decel, currentSpeed, onInsertion, headway, limits);
}

inline double calculateEmergencyDeceleration(double gap, double egoSpeed, double predSpeed, double predMaxDecel, const Limits& limits) {
    if (gap <= 0.) {
        return limits.emergencyDecel;
    }
    const double predBrakeDist = 0.5 * predSpeed * predSpeed / predMaxDecel;
    const double b1 = 0.5 * egoSpeed * egoSpeed / (gap + predBrakeDist);
    return std::max(b1, predMaxDecel);
}

inline double maximumSafeFollowSpeed(double gap, double egoSpeed, double predSpeed, double predMaxDecel, bool onInsertion, const Limits& limits) {
    const double headway = limits.headwayTime;
    double x = 0.;
    if (gap >= 0 || limits.computeLC) {
        x = maximumSafeStopSpeed(gap + brakeGap(predSpeed, std::max(limits.decel, predMaxDecel), 0, limits),
                                 limits.decel, egoSpeed, onInsertion, headway, limits);
    } else {
        x = egoSpeed - accelToSpeed(limits.emergencyDecel, limits.timeStep);
        if (limits.semiImplicitEuler) {
            x = std::max(x, 0.);
        }
    }

    if (limits.decel != limits.emergencyDecel && !onInsertion && !limits.computeLC) {
        const double origSafeDecel = speedToAccel(egoSpeed - x, limits.timeStep);
        if (origSafeDecel > limits.decel + NUMERICAL_EPS) {
            double safeDecel = EMERGENCY_DECEL_AMPLIFIER * calculateEmergencyDeceleration(gap, egoSpeed, predSpeed, predMaxDecel, limits);
            safeDecel = std::max(safeDecel, limits.decel);
            safeDecel = std::min(safeDecel, origSafeDecel);
            x = egoSpeed - accelToSpeed(safeDecel, limits.timeStep);
            if (limits.semiImplicitEuler) {
                x = std::max(x, 0.);
            }
        }
    }
    return x;
}

} // namespace sumo::cfkernel
