#include "TestMnist.h"

#include <iostream>

#include "../Network.h"
#include "../utils.h"

namespace {
nn::Scalar CalculateAccuracy(const nn::Network &net, const nn::Data &test) {
    nn::Index cnt = 0;
    nn::Data shuffle_test = ShuffleData(test);
    for (nn::Index i = 0; i < shuffle_test.X.cols(); ++i) {
        if (nn::ArgMax(net.Predict(shuffle_test.X.col(i))) == nn::ArgMax(shuffle_test.y.col(i))) {
            ++cnt;
        }
    }
    return static_cast<nn::Scalar>(cnt) / test.X.cols();
}
}
void test::TestMnist() {
    nn::Data train =
        nn::mnist::LoadData("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte");
    nn::Data test =
        nn::mnist::LoadData("../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte");

    nn::Network net1({784, 90, 10}, {nn::Sigmoid(), nn::SoftMax()});
    nn::LossFunction loss = nn::MSE();
    nn::LearningRate lr = nn::Constant(3);
    net1.Train(train, 10, 32, loss, lr, test);
    std::cout << "Accuracy: " << CalculateAccuracy(net1, test) << '\n';
}
