#include "MSE.h"
namespace dl {
double MSE::operator()(const Eigen::VectorXd &x, const Eigen::VectorXd &y) const {
    assert(x.size() == y.size());
    return (x - y).dot(x - y);
}

Eigen::RowVectorXd MSE::GetGradient(const Eigen::VectorXd &predicted,
                                    const Eigen::VectorXd &answer) const {
    assert(predicted.size() == answer.size());
    return 2 * (predicted - answer);
}
}  // namespace dl
