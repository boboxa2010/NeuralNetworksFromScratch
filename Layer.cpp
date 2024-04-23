#include "Layer.h"

#include "utils.h"

namespace nn {
Vector Layer::Evaluate(const Vector &x) const {
    assert(x.size() == weights_.cols());
    return function_->Evaluate(weights_ * x + bias_);
}

RowVector Layer::GetNextGradient(const Vector &x, const RowVector &u) const {
    assert(x.size() == weights_.cols());
    assert(u.size() == weights_.rows());
    return u * function_->GetDifferential(weights_ * x + bias_) * weights_;
}

void Layer::Update(LearningRate &learning_rate) {
    Scalar lr = learning_rate();
    weights_ -= lr * grad_weights_;
    bias_ -= lr * grad_bias_;
}

RowVector Layer::BackPropagation(const Vector &x, const RowVector &u) {
    grad_weights_ += GetWeightsGradient(x, u);
    grad_bias_ += GetBiasGradient(x, u);
    return GetNextGradient(x, u);
}

void Layer::ZeroGrad() {
    grad_weights_.setZero();
    grad_bias_.setZero();
}

Matrix Layer::GetWeightsGradient(const Vector &x, const RowVector &u) const {
    return GetBiasGradient(x, u) * x.transpose();
}

Vector Layer::GetBiasGradient(const Vector &x, const RowVector &u) const {
    assert(x.size() == weights_.cols());
    assert(u.size() == weights_.rows());
    return function_->GetDifferential(weights_ * x + bias_) * u.transpose();
}
}  // namespace nn
