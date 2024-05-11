#include "LearningRate.h"

namespace nn {
VowpalWabbit::VowpalWabbit(Scalar lambda, Scalar s0, Scalar power)
    : lambda_(lambda), s0_(s0), power_(power) {
}

Scalar VowpalWabbit::GetValue() {
    ++iteration_;
    return lambda_ * std::pow(s0_ / (s0_ + iteration_), power_);
}

Constant::Constant(Scalar param) : constant_(param) {
}

Scalar Constant::GetValue() {
    return constant_;
}

Gradual::Gradual(Scalar start, Scalar step) : start_(start), step_(step) {
}

Scalar Gradual::GetValue() {
    return start_ + step_ * iteration_++;
}
}  // namespace nn
