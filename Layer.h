#pragma once

#include <memory>

#include "ActivationFunction.h"
#include "Eigen/Eigen"

namespace dl {
class Layer {
public:
    Layer();

    Layer(size_t n, size_t m, std::unique_ptr<ActivationFunction> f);

    Eigen::VectorXd Evaluate(const Eigen::VectorXd &x) const;

    Eigen::VectorXd EvaluateDerivative(const Eigen::VectorXd &x) const;

    Eigen::MatrixXd GetWeightsGradient(const Eigen::VectorXd &x, const Eigen::RowVectorXd &u) const;

    Eigen::VectorXd GetBiasGradient(const Eigen::VectorXd &x, const Eigen::RowVectorXd &u) const;

    Eigen::VectorXd GetNextGradient(const Eigen::VectorXd &x, const Eigen::RowVectorXd &u) const;

    void Update(const Eigen::MatrixXd &weights_grad, const Eigen::VectorXd &bias_grad,
                double learning_rate);

private:
    Eigen::MatrixXd weights_;
    Eigen::VectorXd bias_;
    std::unique_ptr<ActivationFunction> function_;
};
}  // namespace dl