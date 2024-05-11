#include "Tests.h"

#include <iostream>

#include "TestActivationFunction.h"
#include "TestLoss.h"
#include "TestMnist.h"

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

    //MNIST
    TestMnist();

    std::cout << "OK)" << '\n';
}
}  // namespace test
