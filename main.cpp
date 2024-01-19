#include "MnistDataset.h"
#include "utils.h"

int main() {
    dl::Data train = dl::MnistDataset::LoadData("../data/train-images.idx3-ubyte",
                                                "../data/train-labels.idx1-ubyte");
    dl::AsciiRender(train.X[0], train.y[0]);
}
