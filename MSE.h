#pragma once

#include "declarations.h"

namespace nn {
class MSE {
    using FuncT = std::function<Scalar(Scalar, Scalar)>;

public:
    MSE();

    Scalar operator()(const Vector &x, const Vector &y) const noexcept;

    RowVector GetGradient(const Vector &predicted, const Vector &target) const noexcept;

private:
    FuncT derivative_;
};
}  // namespace nn
