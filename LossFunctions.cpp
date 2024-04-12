#include "LossFunctions.h"

namespace nn {
Scalar MSE::operator()(const Vector &x, const Vector &y) const {
    assert(x.size() == y.size());
    return (x - y).dot(x - y);
}

RowVector MSE::GetGradient(const Vector &predicted, const Vector &target) const {
    assert(predicted.size() == target.size());
    return 2 * (predicted - target);
}

Scalar CrossEntropy::operator()(const Vector &x, const Vector &y) const {
    assert(x.size() == y.size());
    return -y.dot(x.unaryExpr([](Scalar x) { return log2(x); }));
}

RowVector CrossEntropy::GetGradient(const Vector &predicted, const Vector &target) const {
    assert(predicted.size() == target.size());
    return target - predicted;
}
}  // namespace nn
