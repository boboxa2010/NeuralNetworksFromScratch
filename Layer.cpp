#include "Layer.h"

#include "EigenRand/EigenRand"

namespace dl {
namespace {
Eigen::Rand::P8_mt19937_64 urng{42};
}

Layer::Layer() = default;

Layer::Layer(size_t n, size_t m, std::unique_ptr<ActivationFunction> f)
    : function_(std::move(f)), weights_(Eigen::MatrixXd{m, n}), bias_(Eigen::VectorXd{m}) {
    weights_ = Eigen::Rand::normalLike(weights_, urng);
    bias_ = Eigen::Rand::normalLike(bias_, urng);
}

Eigen::VectorXd Layer::Evaluate(const Eigen::VectorXd &x) const {
    Eigen::VectorXd result = weights_ * x + bias_;
    function_->ApplyFunction(result.data(), result.data() + result.size());
    return result;
}

Eigen::VectorXd Layer::EvaluateDerivative(const Eigen::VectorXd &x) const {
    Eigen::VectorXd result = weights_ * x + bias_;
    function_->ApplyDerivative(result.data(), result.data() + result.size());
    return result;
}

Eigen::MatrixXd Layer::GetWeightsGradient(const Eigen::VectorXd &x,
                                          const Eigen::RowVectorXd &u) const {
    return GetBiasGradient(x, u) * x.transpose();
}

Eigen::VectorXd Layer::GetBiasGradient(const Eigen::VectorXd &x,
                                       const Eigen::RowVectorXd &u) const {
    return function_->GetDifferential(weights_ * x + bias_) * u.transpose();
}

Eigen::VectorXd Layer::GetNextGradient(const Eigen::VectorXd &x,
                                       const Eigen::RowVectorXd &u) const {
    return u * function_->GetDifferential(weights_ * x + bias_) * weights_;
}

void Layer::Update(const Eigen::MatrixXd &weights_grad, const Eigen::VectorXd &bias_grad,
                   double learning_rate) {
    weights_ -= learning_rate * weights_grad;
    bias_ -= learning_rate * bias_grad;
}
}  // namespace dl