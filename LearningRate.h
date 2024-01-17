#pragma once

#include <cstddef>
#include <cmath>

namespace {
    constexpr double kDefaultLambda = 1e-3;
    constexpr double kDefaultS0 = 1;
    constexpr double kDefaultPower = 0.5;
}

class LearningRate {
public:
    LearningRate();

    LearningRate(double lambda, double s0, double power);

    double operator()();

private:
    double lambda_ = kDefaultLambda;
    double s0_ = kDefaultS0;
    double power_ = kDefaultPower;
    size_t iteration_ = 0;
};
