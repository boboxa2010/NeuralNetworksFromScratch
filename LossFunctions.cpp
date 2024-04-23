#include "LossFunctions.h"
#include "ActivationFunctions.h"
#include <iostream>

namespace nn {
Scalar MSE::operator()(const Matrix &x, const Matrix &y) const {
    assert(x.cols() == y.cols() && x.rows() == y.rows());
    return (x - y).cwiseProduct(x - y).colwise().sum().mean();
}

RowVector MSE::GetGradient(const Matrix &predicted, const Matrix &target) const {
    assert(predicted.cols() == target.cols() && predicted.rows() == target.rows());
    return 2 * (predicted - target).colwise().mean();
}

Scalar CrossEntropy::operator()(const Matrix &x, const Matrix &y) const {
    assert(x.cols() == y.cols() && x.rows() == y.rows());
    return (-y).cwiseProduct(x.unaryExpr([](Scalar x) { return log2(x); })).colwise().sum().mean();
}

RowVector CrossEntropy::GetGradient(const Matrix &predicted, const Matrix &target) const {
    assert(predicted.cols() == target.cols() && predicted.rows() == target.rows());
    return (-target).cwiseProduct(predicted.cwiseInverse()).colwise().mean();
}
}  // namespace nn
