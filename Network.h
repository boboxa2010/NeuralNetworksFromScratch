#pragma once

#include "Layer.h"
#include "LossFunctions.h"
#include "MnistDataset.h"
#include "initializer_list"
namespace nn {
class Network {
public:
    Network(std::initializer_list<Layer> layers);

    Matrix operator()(const Matrix& input) const;

    void Train(const Data& data, Index n_epochs, Index batch_size, const LossFunction& loss,
               LearningRate& lr);

private:
    std::vector<Matrix> ForwardPass(const Matrix& x) const;

    void BackwardPass(const std::vector<Matrix>& predicted, const RowVector& u);

    void ZeroGrad();

    void Step(LearningRate& lr);

    std::vector<Layer> layers_;
};
}  // namespace nn
