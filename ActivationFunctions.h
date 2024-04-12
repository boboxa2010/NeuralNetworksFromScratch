#pragma once

#include <algorithm>
#include <functional>

#include "Any.h"
#include "declarations.h"

namespace nn {
namespace activation_function {
template <typename Base>
class Interface : public Base {
public:
    virtual Vector ApplyFunction(const Vector &v) const = 0;

    virtual Vector ApplyDerivative(const Vector &v) const = 0;

    virtual Matrix GetDifferential(const Vector &v) const = 0;
};

template <typename Base>
class Impl : public Base {
public:
    using Base::Base;

    Vector ApplyFunction(const Vector &v) const {
        return Base::Get().ApplyFunction(v);
    }

    Vector ApplyDerivative(const Vector &v) const {
        return Base::Get().ApplyDerivative(v);
    }

    Matrix GetDifferential(const Vector &v) const {
        return Base::Get().GetDifferential(v);
    }
};
}  // namespace activation_function

using ActivationFunction = Any<activation_function::Interface, activation_function::Impl>;

class CoordinateFunction {
    using FuncT = std::function<Scalar(Scalar)>;

public:
    CoordinateFunction() = default;

    template <class F1, class F2>
    CoordinateFunction(F1 &&function, F2 &&derivative)
        : function_(std::forward<F1>(function)), derivative_(std::forward<F2>(derivative)) {
    }

    Scalar ApplyFunction(Scalar x) const;

    Scalar ApplyDerivative(Scalar x) const;

    template <typename IterType>
    void ApplyFunction(IterType first, IterType last) const {
        std::for_each(first, last, [this](Scalar &x) { x = ApplyFunction(x); });
    }

    template <typename IterType>
    void ApplyDerivative(IterType first, IterType last) const {
        std::for_each(first, last, [this](Scalar &x) { x = ApplyDerivative(x); });
    }

    Vector ApplyFunction(const Vector &v) const;

    Vector ApplyDerivative(const Vector &v) const;

    Matrix GetDifferential(const Vector &v) const;

    operator bool() const;

private:
    FuncT function_;
    FuncT derivative_;
};

class SigmoidFunction : public CoordinateFunction {
public:
    SigmoidFunction();
};

class ReLuFunction : public CoordinateFunction {
public:
    ReLuFunction();
};

class LinearFunction : public CoordinateFunction {
public:
    LinearFunction();
};

class SoftMax {
public:
    SoftMax() = default;

    Vector ApplyFunction(const Vector &v) const;

    Vector ApplyDerivative(const Vector &v) const;

    Matrix GetDifferential(const Vector &v) const;
};
}  // namespace nn
