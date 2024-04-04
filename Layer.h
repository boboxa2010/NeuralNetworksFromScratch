#pragma once

#include "ActivationFunctions.h"
#include "LearningRate.h"

namespace nn {
struct Input {
    explicit Input(size_t size);

    size_t size;
};
struct Output {
    explicit Output(size_t size);

    size_t size;
};
class Layer {
public:
    Layer();

    Layer(Input input_size, Output output_size, const ActivationFunction &f);

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
    ActivationFunction function_;
};
}  // namespace nn
