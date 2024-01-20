#include "Layer.h"

#include "EigenRand/EigenRand"

namespace project {
namespace {
RandomSeed urng = 42;
}

Layer::Layer() = default;

Layer::Layer(size_t input_size, size_t output_size, std::unique_ptr<ActivationFunction> f)
    : function_(std::move(f)),
      weights_(Matrix{output_size, input_size}),
      bias_(Vector{output_size}) {
    weights_ = Eigen::Rand::normalLike(weights_, urng);
    bias_ = Eigen::Rand::normalLike(bias_, urng);
}

Eigen::VectorXd Layer::Evaluate(const Vector &x) const noexcept {
    Eigen::VectorXd result = weights_ * x + bias_;
    function_->ApplyFunction(result.data(), result.data() + result.size());
    return result;
}

Eigen::VectorXd Layer::EvaluateDerivative(const Vector &x) const noexcept {
    Eigen::VectorXd result = weights_ * x + bias_;
    function_->ApplyDerivative(result.data(), result.data() + result.size());
    return result;
}

Eigen::MatrixXd Layer::GetWeightsGradient(const Vector &x, const RowVector &u) const noexcept {
    return GetBiasGradient(x, u) * x.transpose();
}

Eigen::VectorXd Layer::GetBiasGradient(const Vector &x, const RowVector &u) const noexcept {
    return function_->GetDifferential(weights_ * x + bias_) * u.transpose();
}

Eigen::VectorXd Layer::GetNextGradient(const Vector &x, const RowVector &u) const noexcept {
    return u * function_->GetDifferential(weights_ * x + bias_) * weights_;
}

void Layer::Update(const Matrix &weights_grad, const Vector &bias_grad,
                   LearningRate &learning_rate) noexcept {
    weights_ -= learning_rate() * weights_grad;
    bias_ -= learning_rate() * bias_grad;
}
}  // namespace project