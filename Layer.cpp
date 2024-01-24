#include "Layer.h"

#include "utils.h"

namespace nn {
constexpr Input::Input(size_t size) : size(size) {
}

constexpr Output::Output(size_t size) : size(size) {
}

Layer::Layer() = default;

Layer::Layer(Input input, Output output, std::unique_ptr<ActivationFunction> f)
    : function_(std::move(f)),
      weights_(GenerateRandNMatrix(output.size, input.size)),
      bias_(GenerateRandNVector(output.size)) {
}

Vector Layer::Evaluate(const Vector &x) const noexcept {
    assert(x.size() == weights_.cols());
    Vector result = weights_ * x + bias_;
    function_->ApplyFunction(result.data(), result.data() + result.size());
    return result;
}

Vector Layer::EvaluateDerivative(const Vector &x) const noexcept {
    assert(x.size() == weights_.cols());
    Vector result = weights_ * x + bias_;
    function_->ApplyDerivative(result.data(), result.data() + result.size());
    return result;
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
    weights_ -= learning_rate() * weights_grad;
    bias_ -= learning_rate() * bias_grad;
}
}  // namespace nn
