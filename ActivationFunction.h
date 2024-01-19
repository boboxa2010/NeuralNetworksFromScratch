#pragma once

#include <algorithm>
#include <functional>

#include "Eigen/Eigen"

namespace dl {
class ActivationFunction {
    using FuncT = std::function<double(double)>;

public:
    ActivationFunction(const FuncT &function, const FuncT &derivative);

    double ApplyFunction(double x) const;

    double ApplyDerivative(double x) const;

    template <typename IterType>
    void ApplyFunction(IterType first, IterType last) const {
        std::for_each(first, last, [this](double &x) { x = function_(x); });
    }

    template <typename IterType>
    void ApplyDerivative(IterType first, IterType last) const {
        std::for_each(first, last, [this](double &x) { x = derivative_(x); });
    }

    Eigen::MatrixXd GetDifferential(const Eigen::VectorXd &v) const;

private:
    FuncT function_;
    FuncT derivative_;
};

class SigmoidFunction : public ActivationFunction {
public:
    SigmoidFunction();
};

class ReLuFunction : public ActivationFunction {
public:
    ReLuFunction();
};

class LinearFunction : public ActivationFunction {
public:
    LinearFunction();
};

}  // namespace dl
