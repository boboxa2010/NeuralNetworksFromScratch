#pragma once

#include "declarations.h"

namespace project {
    class MSE {
        using FuncT = std::function<NumT(NumT, NumT)>;
    public:
        MSE();

        NumT operator()(const Vector &x, const Vector &y) const noexcept;

        RowVector GetGradient(const Vector &predicted,
                              const Vector &target) const noexcept;

    private:
        FuncT derivative_;
    };
}  // namespace dl
