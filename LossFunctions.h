#pragma once

#include "Any.h"
#include "declarations.h"

namespace nn {
namespace details {
template <typename Base>
class ILossFunction : public Base {
public:
    virtual Scalar operator()(const Matrix &x, const Matrix &y) const = 0;

    virtual RowVector GetGradient(const Matrix &predicted, const Matrix &target) const = 0;
};

template <typename Base>
class ImplLossFunction : public Base {
public:
    using Base::Base;

    Scalar operator()(const Matrix &x, const Matrix &y) const {
        return Base::Get().operator()(x, y);
    }

    RowVector GetGradient(const Matrix &predicted, const Matrix &target) const {
        return Base::Get().GetGradient(predicted, target);
    }
};
}  // namespace details
using LossFunction = Any<details::ILossFunction, details::ImplLossFunction>;

class MSE {
public:
    Scalar operator()(const Matrix &x, const Matrix &y) const;

    RowVector GetGradient(const Matrix &predicted, const Matrix &target) const;
};

class CrossEntropy {
public:
    Scalar operator()(const Matrix &x, const Matrix &y) const;

    RowVector GetGradient(const Matrix &predicted, const Matrix &target) const;
};
}  // namespace nn
