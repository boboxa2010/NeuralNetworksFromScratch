#pragma once

#include "declarations.h"

namespace project {
class LearningRate {
public:
    LearningRate();

    LearningRate(NumT lambda, NumT s0, NumT power);

    NumT operator()();

private:
    static constexpr NumT kDefaultLambda = 1e-3;
    static constexpr NumT kDefaultS0 = 1;
    static constexpr NumT kDefaultPower = 0.5;

    NumT lambda_ = kDefaultLambda;
    NumT s0_ = kDefaultS0;
    NumT power_ = kDefaultPower;
    size_t iteration_{0};
};
}  // namespace project