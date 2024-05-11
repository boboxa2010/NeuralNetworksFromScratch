#include "TestLoss.h"

#include "../LossFunctions.h"

namespace {
class ExampleLoss {
public:
    nn::Vector Evaluate(const nn::Matrix &x, const nn::Matrix &y) const {
        return nn::Vector::Zero(x.cols());
    }

    nn::Matrix GetGradient(const nn::Matrix &predicted, const nn::Matrix &target) const {
        return nn::Matrix::Zero(target.rows(), target.cols()).transpose();
    }
};
}  // namespace

namespace test {
void TestMSE() {
    nn::MSE loss;

    nn::Matrix x{2, 2};
    x << 1, 2, 2, 1;
    nn::Matrix y{2, 2};
    y << 2, 1, 1, 2;

    nn::Vector evaluated{2};
    evaluated << 2, 2;
    assert(loss.Evaluate(x, y).isApprox(evaluated));

    nn::Matrix grads{2, 2};
    grads << -2, 2, 2, -2;
    assert(loss.GetGradient(x, y).isApprox(grads));
}

void TestCrossEntropy() {
    nn::CrossEntropy loss;

    nn::Matrix x{2, 2};
    x << 1, 2, 2, 1;
    nn::Matrix y{2, 2};
    y << 2, 1, 1, 2;

    nn::Vector evaluated{2};
    evaluated << -1, -1;
    assert(loss.Evaluate(x, y).isApprox(evaluated, 1e-3));

    nn::Matrix grads{2, 2};
    grads << -2, -0.5, -0.5, -2;
    assert(loss.GetGradient(x, y).isApprox(grads, 1e-3));
}

void TestNewLoss() {
    nn::LossFunction loss = ExampleLoss();

    nn::Matrix x{2, 2};
    x << 1, 2, 2, 1;
    nn::Matrix y{2, 2};
    y << 2, 1, 1, 2;

    assert(loss->Evaluate(x, y).isZero());
    assert(loss->GetGradient(x, y).isZero());
}

void TestLossCorrectness() {
    nn::LossFunction loss = nn::MSE();
    nn::MSE mse;

    nn::Matrix x{2, 2};
    x << 1, 2, 2, 1;
    nn::Matrix y{2, 2};
    y << 2, 1, 1, 2;

    assert(loss->Evaluate(x, y).isApprox(mse.Evaluate(x, y)));
    assert(loss->GetGradient(x, y).isApprox(mse.GetGradient(x, y)));
}
}  // namespace test
