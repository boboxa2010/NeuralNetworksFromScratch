#pragma once

#include "Any.h"
#include "declarations.h"

namespace nn {
namespace details {
template <typename Base>
class ILossFunction : public Base {
public:
    virtual Scalar operator()(const Vector &x, const Vector &y) const = 0;

    virtual RowVector GetGradient(const Vector &predicted, const Vector &target) const = 0;
};

template <typename Base>
class ImplLossFunction : public Base {
public:
    using Base::Base;

    Scalar operator()(const Vector &x, const Vector &y) const {
        return Base::Get().operator()(x, y);
    }

    RowVector GetGradient(const Vector &predicted, const Vector &target) const {
        return Base::Get().GetGradient(predicted, target);
    }
};
}  // namespace details
using LossFunction = Any<details::ILossFunction, details::ImplLossFunction>;

class MSE {
public:
    Scalar operator()(const Vector &x, const Vector &y) const;

    RowVector GetGradient(const Vector &predicted, const Vector &target) const;
};

class CrossEntropy {
public:
    Scalar operator()(const Vector &x, const Vector &y) const;

    RowVector GetGradient(const Vector &predicted, const Vector &target) const;
};
}  // namespace nn
