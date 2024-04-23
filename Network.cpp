#include "Network.h"

#include <iostream>

namespace nn {
Network::Network(std::initializer_list<Layer> layers) {
    //TODO ASSERT FOR DIM
    layers_.reserve(layers.size());
    for (auto it = layers.begin(); it != layers.end(); ++it) {
        layers_.emplace_back(*it);
    }
}

Matrix Network::operator()(const Matrix &input) const {
    Matrix output = input;
    for (const auto &layer : layers_) {
        output = layer.Evaluate(output);
    }
    return output;
}

void Network::Train(const Data &data, Index n_epochs, Index batch_size, const LossFunction &loss,
                    LearningRate &lr) {
    for (Index epoch = 1; epoch <= n_epochs; ++epoch) {
        for (Index i = 0; i < data.X.cols(); ++i) {
            ZeroGrad();
            auto x_batch = data.X.col(i);
            auto y_batch = data.y.col(i);
            auto predicted = ForwardPass(x_batch);

            auto grad = loss->GetGradient(predicted.back(), y_batch);

            BackwardPass(predicted, grad);

            Step(lr);
        }
    }
}

std::vector<Matrix> Network::ForwardPass(const Matrix &x) const {
    std::vector<Matrix> res;
    res.reserve(layers_.size() + 1);
    res.emplace_back(x);
    Matrix output = x;
    for (const auto &layer : layers_) {
        output = layer.Evaluate(output);
        res.emplace_back(output);
    }
    return res;
}

void Network::ZeroGrad()  {
    std::for_each(layers_.begin(), layers_.end(), [](Layer& layer) { layer.ZeroGrad(); });
}

void Network::BackwardPass(const std::vector<Matrix>& predicted, const RowVector &u){
    auto current_grad = u;
    for (Index i = layers_.size() - 1; i >= 0; --i) {
        current_grad = layers_[i].BackPropagation(predicted[i], current_grad);
    }
}

void Network::Step(LearningRate& lr) {
    std::for_each(layers_.begin(), layers_.end(), [&lr](Layer& layer) { layer.Update(lr); });
}
}  // namespace nn
