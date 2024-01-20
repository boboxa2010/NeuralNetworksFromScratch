#include "LearningRate.h"

namespace project {

LearningRate::LearningRate() = default;

LearningRate::LearningRate(NumT lambda, NumT s0, NumT power)
    : lambda_(lambda), s0_(s0), power_(power) {
}

NumT LearningRate::operator()() {
    ++iteration_;
    return lambda_ * std::pow(s0_ / (s0_ + iteration_), power_);
}

}  // namespace project
