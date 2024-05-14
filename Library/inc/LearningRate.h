#pragma once

#include "Any.h"
#include "declarations.h"

namespace nn {
namespace learning_rate {
template <typename Base>
class ILearningRate : public Base {
public:
    virtual Scalar GetValue() = 0;
};

template <typename Base>
class ImplLearningRate : public Base {
public:
    using Base::Base;

    Scalar GetValue() {
        return Base::Get().GetValue();
    }
};
}  // namespace learning_rate

using LearningRate = Any<learning_rate::ILearningRate, learning_rate::ImplLearningRate>;

class VowpalWabbit {
public:
    VowpalWabbit() = default;

    VowpalWabbit(Scalar lambda, Scalar s0, Scalar power);

    Scalar GetValue();

private:
    static constexpr Scalar kDefaultLambda = 1e-3;
    static constexpr Scalar kDefaultS0 = 1;
    static constexpr Scalar kDefaultPower = 0.5;

    Scalar lambda_ = kDefaultLambda;
    Scalar s0_ = kDefaultS0;
    Scalar power_ = kDefaultPower;
    Index iteration_{0};
};

class Constant {
public:
    Constant(Scalar param);

    Scalar GetValue();

private:
    Scalar constant_{0};
};

class Gradual {
public:
    Gradual(Scalar start, Scalar step);

    Scalar GetValue();

private:
    static constexpr Scalar kDefaultStart = 1e-2;
    static constexpr Scalar kDefaultStep = 0.1;

    Scalar start_{kDefaultStart};
    Scalar step_{kDefaultStep};
    Index iteration_{0};
};
}  // namespace nn
