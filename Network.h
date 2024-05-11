#pragma once

#include "Layer.h"
#include "LossFunctions.h"
#include "MnistDataset.h"
#include "initializer_list"

namespace nn {
class Network {
public:
    Network(std::initializer_list<Layer> layers);

    Network(std::initializer_list<Index> dimensions,
            std::initializer_list<ActivationFunction> functions);

    Matrix Predict(const Matrix& input) const;

    void Train(const Data& train, Index n_epochs, Index batch_size, const LossFunction& loss,
               LearningRate& lr, const Data& test);

private:
    void TrainEpoch(const Data& train, Index batch_size, const LossFunction& loss, LearningRate& lr,
                    const Data& test);

    void TrainBatch(const Matrix& x, const Matrix& y, Index batch_size, const LossFunction& loss,
                    LearningRate& lr);

    TenZor ForwardPass(const Matrix& x) const;

    void BackwardPass(const TenZor& predicted, const Matrix& loss_grad);

    void ZeroGrad();

    void Step(LearningRate& lr, Index batch_size);

    std::vector<Layer> layers_;
};
}  // namespace nn
