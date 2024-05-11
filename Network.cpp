#include "Network.h"

#include <iostream>

namespace nn {
Network::Network(std::initializer_list<Layer> layers) {
    for (auto it = layers.begin(); it != layers.end(); ++it) {
        layers_.emplace_back(*it);
    }
}

Network::Network(std::initializer_list<Index> dimensions,
                 std::initializer_list<ActivationFunction> functions) {
    layers_.reserve(functions.size());
    auto dim = dimensions.begin();
    for (auto it = functions.begin(); it != functions.end(); ++it, ++dim) {
        layers_.emplace_back(Input(*dim), Output(*std::next(dim)),
                             std::move(ActivationFunction(*it)));
    }
}

Matrix Network::Predict(const Matrix &input) const {
    Matrix output = input;
    for (const auto &layer : layers_) {
        output = layer.Evaluate(output);
    }
    return output;
}

void Network::Train(const Data &train, Index n_epochs, Index batch_size, const LossFunction &loss,
                    LearningRate &lr, const Data &test) {
    for (Index epoch = 1; epoch <= n_epochs; ++epoch) {
        TrainEpoch(train, batch_size, loss, lr, test);
    }
}

void Network::TrainEpoch(const Data &train, Index batch_size, const LossFunction &loss,
                         LearningRate &lr, const Data &test) {
    Data shuffle_data = ShuffleData(train);
    for (Index i = 0; i < shuffle_data.X.cols(); i += batch_size) {
        const Matrix &x_batch =
            shuffle_data.X.middleCols(i, std::min(batch_size, shuffle_data.X.cols() - i));
        const Matrix &y_batch =
            shuffle_data.y.middleCols(i, std::min(batch_size, shuffle_data.X.cols() - i));
        TrainBatch(x_batch, y_batch, batch_size, loss, lr);
    }
}

void Network::TrainBatch(const Matrix &x, const Matrix &y, Index batch_size,
                         const LossFunction &loss, LearningRate &lr) {
    ZeroGrad();
    auto predicted = ForwardPass(x);
    auto grad = loss->GetGradient(predicted.back(), y);
    BackwardPass(predicted, grad);
    Step(lr, batch_size);
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

void Network::BackwardPass(const TenZor &predicted, const Matrix &loss_grad) {
    auto current_grad = loss_grad;
    for (Index i = layers_.size() - 1; i >= 0; --i) {
        current_grad = layers_[i].BackPropagation(predicted[i], current_grad);
    }
}

void Network::ZeroGrad() {
    std::for_each(layers_.begin(), layers_.end(), [](Layer &layer) { layer.ZeroGrad(); });
}

void Network::Step(LearningRate &learning_rate, Index batch_size) {
    Scalar lr = learning_rate->GetValue();
    std::for_each(layers_.begin(), layers_.end(),
                  [lr, batch_size](Layer &layer) { layer.Update(lr, batch_size); });
}
}  // namespace nn
