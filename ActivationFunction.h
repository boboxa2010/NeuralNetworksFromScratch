#pragma once

#include <algorithm>
#include <functional>

#include "declarations.h"

namespace project {
class ActivationFunction {
    using FuncT = std::function<NumT(NumT)>;

public:
    ActivationFunction(const FuncT &function, const FuncT &derivative);

    double ApplyFunction(NumT x) const;

    double ApplyDerivative(NumT x) const;

    template <typename IterType>
    void ApplyFunction(IterType first, IterType last) const {
        std::for_each(first, last, [this](NumT &x) { x = function_(x); });
    }

    template <typename IterType>
    void ApplyDerivative(IterType first, IterType last) const {
        std::for_each(first, last, [this](NumT &x) { x = derivative_(x); });
    }

    Matrix GetDifferential(const Vector &v) const;

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

}  // namespace project
