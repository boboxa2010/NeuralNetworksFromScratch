#include "MnistDataset.h"
#include "utils.h"

int main() {
    Data train = MnistDataset::LoadData("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte");
    AsciiRender(train.X[0], train.y[0]);
}
