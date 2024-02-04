#include "ActivationFunction.h"

#include <cmath>

namespace nn {
ActivationFunction::ActivationFunction(const FuncT &function, const FuncT &derivative)
    : function_(function), derivative_(derivative) {
}

Scalar ActivationFunction::ApplyFunction(Scalar x) const {
    return function_(x);
}

Scalar ActivationFunction::ApplyDerivative(Scalar x) const {
    return derivative_(x);
}

Matrix ActivationFunction::GetDifferential(const Vector &v) const {
    return v.unaryExpr(derivative_).asDiagonal();
}

SigmoidFunction::SigmoidFunction()
    : ActivationFunction([](Scalar x) { return 1 / (1 + exp(-x)); },
                         [](Scalar x) { return exp(x) / ((1 + exp(x)) * (1 + exp(x))); }) {
}

ReLuFunction::ReLuFunction()
    : ActivationFunction([](Scalar x) { return std::max(x, 0.0); },
                         [](Scalar x) { return x > 0 ? 1 : 0; }) {
}

LinearFunction::LinearFunction()
    : ActivationFunction([](Scalar x) { return x; }, [](Scalar x) { return 1; }) {
}
}  // namespace nn
