#include "../inc/LossFunctions.h"

namespace {
constexpr double kEsp = 1e-5;
}

namespace nn {
Vector MSE::Evaluate(const Matrix &x, const Matrix &y) const {
    assert(x.cols() == y.cols() && x.rows() == y.rows());
    return (x - y).cwiseProduct(x - y).colwise().sum();
}

Matrix MSE::GetGradient(const Matrix &predicted, const Matrix &target) const {
    assert(predicted.cols() == target.cols() && predicted.rows() == target.rows());
    return 2 * (predicted - target).transpose();
}

Vector CrossEntropy::Evaluate(const Matrix &x, const Matrix &y) const {
    assert(x.cols() == y.cols() && x.rows() == y.rows());
    return (-y).cwiseProduct(x.unaryExpr([](Scalar x) { return log2(x + kEsp); })).colwise().sum();
}

Matrix CrossEntropy::GetGradient(const Matrix &predicted, const Matrix &target) const {
    assert(predicted.cols() == target.cols() && predicted.rows() == target.rows());
    return (-target).cwiseProduct((predicted.array() + kEsp).matrix().cwiseInverse()).transpose();
}
}  // namespace nn
