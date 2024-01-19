#include "ActivationFunction.h"

#include <cmath>

namespace dl {
ActivationFunction::ActivationFunction(const FuncT &function, const FuncT &derivative)
    : function_(function), derivative_(derivative) {
}

double ActivationFunction::ApplyFunction(double x) const {
    return function_(x);
}

double ActivationFunction::ApplyDerivative(double x) const {
    return derivative_(x);
}

Eigen::MatrixXd ActivationFunction::GetDifferential(const Eigen::VectorXd &v) const {
    return v.unaryExpr(function_).asDiagonal();
}

SigmoidFunction::SigmoidFunction()
    : ActivationFunction([](double x) { return 1 / (1 + exp(-x)); },
                         [](double x) { return exp(x) / ((1 + exp(x)) * (1 + exp(x))); }) {
}

ReLuFunction::ReLuFunction()
    : ActivationFunction([](double x) { return x > 0 ? x : 0; },
                         [](double x) { return x > 0 ? 1 : 0; }) {
}

LinearFunction::LinearFunction()
    : ActivationFunction([](double x) { return x; }, [](double x) { return 1; }) {
}
}  // namespace dl