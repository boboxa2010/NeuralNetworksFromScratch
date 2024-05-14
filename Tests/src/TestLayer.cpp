#include "../inc/TestLayer.h"

#include "../Library/inc/Layer.h"

void test::TestCreateLayer() {
    nn::Layer layer(nn::Input(10), nn::Output(100), nn::ReLu());

    assert(layer.GetBias().size() == 100);
    assert(layer.GetWeights().rows() == 100 && layer.GetWeights().cols() == 10);
}
