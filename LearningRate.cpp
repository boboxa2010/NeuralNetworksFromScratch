#include "LearningRate.h"

namespace nn {
LearningRate::LearningRate(Scalar lambda, Scalar s0, Scalar power)
    : lambda_(lambda), s0_(s0), power_(power) {
}

Scalar LearningRate::operator()() {
    ++iteration_;
    return lambda_ * std::pow(s0_ / (s0_ + iteration_), power_);
}

}  // namespace nn
