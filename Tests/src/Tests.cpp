#include "../inc/Tests.h"

#include <iostream>

#include "../inc/TestActivationFunction.h"
#include "../inc/TestLoss.h"
#include "../inc/TestMnist.h"
#include "../inc/TestLayer.h"

namespace test {
void RunAllTests() {

    // ActivationFunc
    TestReLuEvaluate();
    TestReLuDerivative();
    TestLeakyReLuEvaluate();
    TestLeakyReLuDerivative();
    TestLinearEvaluate();
    TestLinearDerivative();
    TestSigmoidEvaluate();
    TestSigmoidDerivative();
    TestSoftMaxEvaluate();
    TestSoftMaxDerivative();
    TestNewActivation();
    TestActivationCorrectness();

    // Loss
    TestMSE();
    TestNewLoss();
    TestCrossEntropy();
    TestLossCorrectness();

    // Layer
    TestCreateLayer();
    //MNIST
    TestMnist();

    std::cout << "OK)" << '\n';
}
}  // namespace test
