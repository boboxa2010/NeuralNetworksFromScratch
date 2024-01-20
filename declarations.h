#pragma once

#include "Eigen/Dense"
#include "EigenRand/EigenRand"

namespace project {
using NumT = double;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Matrix = Eigen::MatrixXd;
using RandomSeed = Eigen::Rand::P8_mt19937_64;
}  // namespace project