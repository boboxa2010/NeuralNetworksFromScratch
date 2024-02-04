#pragma once

#include "declarations.h"

namespace nn {
class LearningRate {
public:
    LearningRate();

    LearningRate(Scalar lambda, Scalar s0, Scalar power);

    Scalar operator()();

private:
    static constexpr Scalar kDefaultLambda = 1e-3;
    static constexpr Scalar kDefaultS0 = 1;
    static constexpr Scalar kDefaultPower = 0.5;

    Scalar lambda_ = kDefaultLambda;
    Scalar s0_ = kDefaultS0;
    Scalar power_ = kDefaultPower;
    size_t iteration_{0};
};
}  // namespace nn
