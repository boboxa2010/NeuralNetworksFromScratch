#include "ActivationFunction.h"

#include <cmath>

namespace project {
ActivationFunction::ActivationFunction(const FuncT &function, const FuncT &derivative)
    : function_(function), derivative_(derivative) {
}

NumT ActivationFunction::ApplyFunction(NumT x) const {
    return function_(x);
}

NumT ActivationFunction::ApplyDerivative(NumT x) const {
    return derivative_(x);
}

Matrix ActivationFunction::GetDifferential(const Vector &v) const {
    return v.unaryExpr(function_).asDiagonal();
}

SigmoidFunction::SigmoidFunction()
    : ActivationFunction([](NumT x) { return 1 / (1 + exp(-x)); },
                         [](NumT x) { return exp(x) / ((1 + exp(x)) * (1 + exp(x))); }) {
}

ReLuFunction::ReLuFunction()
    : ActivationFunction([](NumT x) { return x > 0 ? x : 0; },
                         [](NumT x) { return x > 0 ? 1 : 0; }) {
}

LinearFunction::LinearFunction()
    : ActivationFunction([](NumT x) { return x; }, [](NumT x) { return 1; }) {
}
}  // namespace project