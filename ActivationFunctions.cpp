#include "ActivationFunctions.h"

#include <cmath>

namespace nn {
Scalar ElemWiseFunction::Evaluate(Scalar x) const {
    assert(function_);
    return function_(x);
}

Scalar ElemWiseFunction::EvaluateDerivative(Scalar x) const {
    assert(derivative_);
    return derivative_(x);
}

Matrix ElemWiseFunction::GetDifferential(const Vector &v) const {
    assert(derivative_);
    return v.unaryExpr(derivative_).asDiagonal();
}

ElemWiseFunction::operator bool() const {
    return function_ && derivative_;
}

Vector ElemWiseFunction::Evaluate(const Vector &v) const {
    assert(function_);
    return v.unaryExpr(function_);
}

Sigmoid::Sigmoid()
    : ElemWiseFunction(
          [](Scalar x) { return 1 / (1 + exp(-x)); },
          [](Scalar x) { return (1 / (1 + exp(-x)) - 1 / ((1 + exp(-x)) * (1 + exp(-x)))); }) {
}

ReLu::ReLu()
    : ElemWiseFunction([](Scalar x) { return x * (x > 0); }, [](Scalar x) { return x > 0; }) {
}

Linear::Linear() : ElemWiseFunction([](Scalar x) { return x; }, [](Scalar) { return 1; }) {
}

Vector SoftMax::Evaluate(const Vector &v) const {
    Vector exp = (v.array() - v.maxCoeff()).exp();
    return exp.array() / exp.sum();
}

Matrix SoftMax::GetDifferential(const Vector &v) const {
    Vector soft_max = Evaluate(v);
    return Matrix(soft_max.asDiagonal()) - (soft_max * soft_max.transpose());
}
}  // namespace nn
