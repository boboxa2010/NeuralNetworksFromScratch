#include "MnistDataset.h"
#include "utils.h"

int main() {
    project::Data train = project::MnistDataset::LoadData("../data/train-images.idx3-ubyte",
                                                          "../data/train-labels.idx1-ubyte");
    project::AsciiRender(train.X[0], train.y[0]);
}
