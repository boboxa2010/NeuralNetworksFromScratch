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
    virtual Matrix Evaluate(const Matrix &x) const = 0;

    virtual Matrix GetDifferential(const Vector &v) const = 0;
};

template <typename Base>
class Impl : public Base {
public:
    using Base::Base;

    Matrix Evaluate(const Matrix &x) const {
        return Base::Get().Evaluate(x);
    }

    Matrix GetDifferential(const Vector &v) const {
        return Base::Get().GetDifferential(v);
    }
};
}  // namespace activation_function

using ActivationFunction = iternal::Any<activation_function::Interface, activation_function::Impl>;

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

    Matrix Evaluate(const Matrix &v) const;

    Matrix EvaluateDerivative(const Matrix &v) const;

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

class LeakyReLu : public ElemWiseFunction {
public:
    LeakyReLu(Scalar slope);
};

class Linear : public ElemWiseFunction {
public:
    Linear();
};

class SoftMax {
public:
    Vector operator()(const Vector &v) const;

    Matrix Evaluate(const Matrix &v) const;

    Matrix GetDifferential(const Vector &v) const;
};
}  // namespace nn
