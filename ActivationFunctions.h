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
    virtual Vector Evaluate(const Vector &v) const = 0;

    virtual Matrix GetDifferential(const Vector &v) const = 0;
};

template <typename Base>
class Impl : public Base {
public:
    using Base::Base;

    Vector Evaluate(const Vector &v) const {
        return Base::Get().Evaluate(v);
    }

    Matrix GetDifferential(const Vector &v) const {
        return Base::Get().GetDifferential(v);
    }
};
}  // namespace activation_function

using ActivationFunction = Any<activation_function::Interface, activation_function::Impl>;

class ElemWiseFunction {
    using FuncT = std::function<Scalar(Scalar)>;
public:
    template <class F1, class F2>
    ElemWiseFunction(F1 &&function, F2 &&derivative)
        : function_(std::forward<F1>(function)), derivative_(std::forward<F2>(derivative)) {
    }

    Scalar Evaluate(Scalar x) const;

    Scalar EvaluateDerivative(Scalar x) const;

    template <typename IterType>
    void Evaluate(IterType first, IterType last) const {
        std::for_each(first, last, [this](Scalar &x) { x = Evaluate(x); });
    }

    template <typename IterType>
    void EvaluateDerivative(IterType first, IterType last) const {
        std::for_each(first, last, [this](Scalar &x) { x = EvaluateDerivative(x); });
    }

    Vector Evaluate(const Vector &v) const;

    Matrix GetDifferential(const Vector &v) const;

    operator bool() const;

private:
    FuncT function_;
    FuncT derivative_;
};

class Sigmoid : public ElemWiseFunction {
public:
    Sigmoid();
};

class ReLu : public ElemWiseFunction {
public:
    ReLu();
};

class Linear : public ElemWiseFunction {
public:
    Linear();
};

class SoftMax {
public:
    SoftMax() = default;

    Vector Evaluate(const Vector &v) const;

    Matrix GetDifferential(const Vector &v) const;
};
}  // namespace nn
