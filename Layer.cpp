#include "Layer.h"

#include "utils.h"

namespace nn {
Layer::Layer(Input input, Output output, ActivationFunction f)
    : function_(std::move(f)),
      weights_(GenerateRandNMatrix(output.size, input.size)),
      bias_(GenerateRandNVector(output.size)) {
}

Layer::Layer(Input input, Output output, const ActivationFunction &f)
    : function_(f),
      weights_(GenerateRandNMatrix(output.size, input.size)),
      bias_(GenerateRandNVector(output.size)) {
}

Vector Layer::Evaluate(const Vector &x) const noexcept {
    assert(x.size() == weights_.cols());
    return function_->ApplyFunction(weights_ * x + bias_);
}

Vector Layer::EvaluateDerivative(const Vector &x) const noexcept {
    assert(x.size() == weights_.cols());
    return function_->ApplyDerivative(weights_ * x + bias_);
}

Matrix Layer::GetWeightsGradient(const Vector &x, const RowVector &u) const noexcept {
    return GetBiasGradient(x, u) * x.transpose();
}

Vector Layer::GetBiasGradient(const Vector &x, const RowVector &u) const noexcept {
    assert(x.size() == weights_.cols());
    assert(u.size() == weights_.rows());
    return function_->GetDifferential(weights_ * x + bias_) * u.transpose();
}

Vector Layer::GetNextGradient(const Vector &x, const RowVector &u) const noexcept {
    assert(x.size() == weights_.cols());
    assert(u.size() == weights_.rows());
    return u * function_->GetDifferential(weights_ * x + bias_) * weights_;
}

void Layer::Update(const Matrix &weights_grad, const Vector &bias_grad,
                   LearningRate &learning_rate) noexcept {
    assert(weights_grad.rows() == weights_.rows() && weights_grad.cols() == weights_.cols());
    assert(bias_grad.size() == bias_.size());
    Scalar lr = learning_rate();
    weights_ -= lr * weights_grad;
    bias_ -= lr * bias_grad;
}
}  // namespace nn
