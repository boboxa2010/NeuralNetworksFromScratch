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
