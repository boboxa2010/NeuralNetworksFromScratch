#include "ActivationFunctions.h"

#include <cmath>

namespace nn {
Scalar CoordinateFunction::ApplyFunction(Scalar x) const {
    assert(function_);
    return function_(x);
}

Scalar CoordinateFunction::ApplyDerivative(Scalar x) const {
    assert(derivative_);
    return derivative_(x);
}

Matrix CoordinateFunction::GetDifferential(const Vector &v) const {
    return v.unaryExpr(derivative_).asDiagonal();
}

CoordinateFunction::operator bool() const {
    return function_ && derivative_;
}

Vector CoordinateFunction::ApplyFunction(const Vector &v) const {
    assert(function_);
    return v.unaryExpr(function_);
}

Vector CoordinateFunction::ApplyDerivative(const Vector &v) const {
    assert(derivative_);
    return v.unaryExpr(derivative_);
}

SigmoidFunction::SigmoidFunction()
    : CoordinateFunction(
          [](Scalar x) { return 1 / (1 + exp(-x)); },
          [](Scalar x) { return (1 / (1 + exp(-x)) - 1 / ((1 + exp(-x)) * (1 + exp(-x)))); }) {
}

ReLuFunction::ReLuFunction()
    : CoordinateFunction([](Scalar x) { return x * (x > 0); }, [](Scalar x) { return x > 0; }) {
}

LinearFunction::LinearFunction()
    : CoordinateFunction([](Scalar x) { return x; }, [](Scalar) { return 1; }) {
}

Matrix SoftMax::GetDifferential(const Vector &v) const {
    Vector soft_max = ApplyFunction(v);
    Matrix differential{v.size(), v.size()};
    for (size_t i = 0; i < differential.cols(); ++i) {
        for (size_t j = 0; j < differential.rows(); ++j) {
            differential(i, j) = soft_max[j] * ((i == j) - soft_max[i]);
        }
    }
    return differential;
}

Vector SoftMax::ApplyFunction(const Vector &v) const {
    Vector exp = (v.array() - v.maxCoeff()).exp();
    return exp.array() / exp.sum();
}

Vector SoftMax::ApplyDerivative(const Vector &v) const {
    return nn::Vector();
}
}  // namespace nn
