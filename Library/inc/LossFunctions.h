#pragma once

#include "Any.h"
#include "declarations.h"

namespace nn {
namespace details {
template <typename Base>
class ILossFunction : public Base {
public:
    virtual Vector Evaluate(const Matrix& x, const Matrix &y) const = 0;

    virtual Matrix GetGradient(const Matrix &predicted, const Matrix &target) const = 0;
};

template <typename Base>
class ImplLossFunction : public Base {
public:
    using Base::Base;

    Vector Evaluate(const Matrix& x, const Matrix &y) const {
        return Base::Get().Evaluate(x, y);
    }

    Matrix GetGradient(const Matrix &predicted, const Matrix &target) const {
        return Base::Get().GetGradient(predicted, target);
    }
};
}  // namespace details
using LossFunction = iternal::Any<details::ILossFunction, details::ImplLossFunction>;

class MSE {
public:
    Vector Evaluate(const Matrix& x, const Matrix &y) const;

    Matrix GetGradient(const Matrix &predicted, const Matrix &target) const;
};

class CrossEntropy {
public:
    Vector Evaluate(const Matrix& x, const Matrix &y) const;

    Matrix GetGradient(const Matrix &predicted, const Matrix &target) const;
};
}  // namespace nn
