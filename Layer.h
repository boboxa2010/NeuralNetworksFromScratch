#pragma once

#include "ActivationFunctions.h"
#include "LearningRate.h"
#include "utils.h"

namespace nn {
struct Input {
    explicit Input(Index size) : size(size) {
    }
    Index size;
};

struct Output {
    explicit Output(Index size) : size(size) {
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

    Matrix Evaluate(const Matrix &x) const;

    Matrix BackPropagation(const Matrix &x, const Matrix &u);

    void Update(Scalar lr, Index batch_size);

    void ZeroGrad();

private:
    Matrix GetNextGradient(const Matrix &delta) const;

    Matrix GetWeightsGradientBatch(const Matrix &x, const Matrix &delta) const;

    Vector GetBiasGradientBatch(const Matrix &delta) const;

    Matrix GetDelta(const Matrix &x, const Matrix &u) const;

    ActivationFunction function_;
    Matrix weights_;
    Vector bias_;
    Matrix grad_weights_;
    Vector grad_bias_;
};
}  // namespace nn
