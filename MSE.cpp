#include "MSE.h"

double MSE::operator()(const Eigen::VectorXd &x, const Eigen::VectorXd &y) const {
    assert(x.size() == y.size());
    return (x - y).dot(x - y);
}

Eigen::RowVectorXd MSE::GetGradient(const Eigen::VectorXd &predicted, const Eigen::VectorXd &answer) const {
    assert(predicted.size() == answer.size());
    return 2 * (predicted - answer);
}

#pragma once

#include "Eigen/Eigen"

class MSE {
public:
    MSE() = default;

    double operator()(const Eigen::VectorXd &x, const Eigen::VectorXd &y) const;

    Eigen::RowVectorXd GetGradient(const Eigen::VectorXd &predicted,
                                   const Eigen::VectorXd &answer) const;

private:
    std::function<double(double, double)> derivative_;
};
