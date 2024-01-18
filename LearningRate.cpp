//
// Created by ivan on 17/01/24.
//

#include "LearningRate.h"

LearningRate::LearningRate() = default;

double LearningRate::operator()() {
    ++iteration_;
    return lambda_ * std::pow(s0_ / (s0_ + iteration_), power_);
}

LearningRate::LearningRate(double lambda, double s0, double power)
    : lambda_(lambda), s0_(s0), power_(power) {
}
