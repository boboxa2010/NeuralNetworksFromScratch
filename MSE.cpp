#include "MSE.h"

namespace project {
MSE::MSE() = default;

NumT MSE::operator()(const Vector &x, const Vector &y) const noexcept {
    assert(x.size() == y.size());
    return (x - y).dot(x - y);
}

RowVector MSE::GetGradient(const Vector &predicted, const Vector &target) const noexcept {
    assert(predicted.size() == target.size());
    return 2 * (predicted - target);
}
}  // namespace project
