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

Matrix ElemWiseFunction::Evaluate(const Matrix &v) const {
    assert(function_);
    return v.unaryExpr(function_);
}

Matrix ElemWiseFunction::EvaluateDerivative(const Matrix &v) const {
    assert(derivative_);
    return v.unaryExpr(derivative_);
}

Matrix ElemWiseFunction::GetDifferential(const Vector &v) const {
    assert(derivative_);
    return v.unaryExpr(derivative_).asDiagonal();
}

ElemWiseFunction::operator bool() const {
    return function_ && derivative_;
}

Sigmoid::Sigmoid()
    : ElemWiseFunction([](Scalar x) { return 1 / (1 + exp(-x)); },
                       [](Scalar x) { return 1 / (1 + exp(-x)) * (1 - 1 / (1 + exp(-x))); }) {
}

ReLu::ReLu()
    : ElemWiseFunction([](Scalar x) { return x * (x > 0); }, [](Scalar x) { return x > 0; }) {
}

LeakyReLu::LeakyReLu(Scalar slope)
    : ElemWiseFunction([slope](Scalar x) { return x > 0 ? x : slope * x; },
                       [slope](Scalar x) { return x > 0 ? 1 : slope; }) {
}

Linear::Linear() : ElemWiseFunction([](Scalar x) { return x; }, [](Scalar) { return 1; }) {
}

Vector SoftMax::operator()(const Vector &v) const {
    Vector exp = (v.array() - v.maxCoeff()).exp();
    return exp.array() / exp.sum();
}

Matrix SoftMax::Evaluate(const Matrix &v) const {
    Matrix shifted = (v.rowwise() - v.colwise().maxCoeff()).array().exp().matrix();
    return ((shifted.array() / ((Vector::Ones(v.rows()) * shifted.colwise().sum()).array())))
        .matrix();
    /*
    return (((v.rowwise() - v.colwise().maxCoeff()).array().exp() /
             ((Vector::Ones(v.rows()) *
               (v.rowwise() - v.colwise().maxCoeff()).array().exp().matrix().colwise().sum())
                  .array())))
        .matrix();
    */
    // Как лучше ? Работают оба правильно вроде
}

Matrix SoftMax::GetDifferential(const Vector &v) const {
    Vector soft_max = operator()(v);
    return Matrix(soft_max.asDiagonal()) - (soft_max * soft_max.transpose());
}
}  // namespace nn
