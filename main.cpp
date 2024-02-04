#include "ActivationFunction.h"
#include "Layer.h"
#include "MnistDataset.h"
#include "utils.h"
int main() {
    nn::mnist::Data train =
        nn::mnist::LoadData("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte");
    nn::AsciiRender(train.X[0], train.y[0]);
}
