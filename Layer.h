#pragma once

#include <memory>

#include "ActivationFunction.h"
#include "LearningRate.h"

namespace project {
class Layer {
public:
    Layer();

    Layer(size_t input_size, size_t output_size, std::unique_ptr<ActivationFunction> f);

    Vector Evaluate(const Vector &x) const noexcept;

    Vector EvaluateDerivative(const Vector &x) const noexcept;

    Matrix GetWeightsGradient(const Vector &x, const RowVector &u) const noexcept;

    Vector GetBiasGradient(const Vector &x, const RowVector &u) const noexcept;

    Vector GetNextGradient(const Vector &x, const RowVector &u) const noexcept;

    void Update(const Matrix &weights_grad, const Vector &bias_grad,
                LearningRate &learning_rate) noexcept;

private:
    Matrix weights_;
    Vector bias_;
    std::unique_ptr<ActivationFunction> function_;
};
}  // namespace project