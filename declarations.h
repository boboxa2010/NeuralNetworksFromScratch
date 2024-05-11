#pragma once

#include "Eigen/Dense"

namespace nn {
using Index = Eigen::Index;
using Scalar = double;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Matrix = Eigen::MatrixXd;
using TenZor = std::vector<Matrix>;
}  // namespace nn
