#pragma once

#include "ActivationFunctions.h"
#include "LearningRate.h"
#include "utils.h"

namespace nn {
struct Input {
    explicit Input(size_t size) : size(size) {
    }
    Index size;
};

struct Output {
    explicit Output(size_t size) : size(size) {
    }
    Index size;
};

class Layer {
public:
    template <typename T>
    Layer(Input input, Output output, T &&f)
        : function_(std::move(f)),
          weights_(GenerateRandNMatrix(output.size, input.size)),
          bias_(GenerateRandNVector(output.size)),
          grad_weights_(Matrix::Zero(output.size, input.size)),
          grad_bias_(Matrix::Zero(output.size, 1)) {
    }

    Vector Evaluate(const Vector &x) const;

    RowVector GetNextGradient(const Vector &x, const RowVector &u) const;

    void Update(LearningRate &learning_rate);

    RowVector BackPropagation(const Vector &x, const RowVector &u);

    void ZeroGrad();

private:
    Matrix GetWeightsGradient(const Vector &x, const RowVector &u) const;

    Vector GetBiasGradient(const Vector &x, const RowVector &u) const;

    Matrix weights_;
    Vector bias_;
    Matrix grad_weights_;
    Vector grad_bias_;
    ActivationFunction function_;
};
}  // namespace nn
