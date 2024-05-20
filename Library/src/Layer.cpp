#include "../inc/Layer.h"

namespace nn {
Matrix Layer::Evaluate(const Matrix &x) const {
    assert(x.rows() == weights_.cols());
    return function_->Evaluate((weights_ * x).colwise() + bias_);
}

Matrix Layer::BackPropagation(const Matrix &x, const Matrix &u) {
    auto delta = GetDelta(x, u);
    grad_bias_ += GetBiasGradientBatch(delta);
    grad_weights_ += GetWeightsGradientBatch(x, delta);
    return GetNextGradient(delta);
}

void Layer::Update(Scalar lr, Index batch_size) {
    assert(batch_size != 0);
    weights_ -= lr * (grad_weights_ / batch_size);
    bias_ -= lr * (grad_bias_ / batch_size);
}

void Layer::ZeroGrad() {
    grad_weights_.setZero();
    grad_bias_.setZero();
}

Matrix Layer::GetWeights() const {
    return weights_;
}

Vector Layer::GetBias() const {
    return bias_;
}

Matrix Layer::GetNextGradient(const Matrix &delta) const {
    assert(delta.rows() == weights_.rows());
    return delta.transpose() * weights_;
}

Matrix Layer::GetWeightsGradientBatch(const Matrix &x, const Matrix &delta) const {
    assert(delta.rows() == weights_.rows() && x.rows() == weights_.cols());
    Matrix weights_grad = Matrix::Zero(weights_.rows(), weights_.cols());
    for (Index i = 0; i < x.cols(); ++i) {
        weights_grad += delta.col(i) * x.col(i).transpose();
    }
    return weights_grad;
}

Vector Layer::GetBiasGradientBatch(const Matrix &delta) const {
    assert(delta.rows() == bias_.size());
    return delta.rowwise().sum();
}

Matrix Layer::GetDelta(const Matrix &x, const Matrix &u) const {
    assert(x.rows() == weights_.cols());
    assert(u.cols() == weights_.rows());
    Matrix delta = Matrix::Zero(bias_.size(), x.cols());
    Matrix z = (weights_ * x).colwise() + bias_;
    for (Index i = 0; i < x.cols(); ++i) {
        delta.col(i) = function_->GetDifferential(z.col(i)) * u.row(i).transpose();
    }
    return delta;
}
}  // namespace nn
