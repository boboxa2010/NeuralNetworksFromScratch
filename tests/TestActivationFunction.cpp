#include "TestActivationFunction.h"

#include <deque>

#include "../ActivationFunctions.h"

namespace {
const double kEps = 1e-5;

template <typename IterType>
bool IsApproxEqual(IterType lhs_begin, IterType lhs_end, IterType rhs_begin) {
    return std::equal(lhs_begin, lhs_end, rhs_begin, [](double value1, double value2) {
        return std::fabs(value1 - value2) < kEps;
    });
}

class ExampleActivation {
public:
    nn::Matrix Evaluate(const nn::Matrix &v) const {
        return nn::Matrix::Zero(v.rows(), v.cols());
    }

    nn::Matrix GetDifferential(const nn::Vector &v) const {
        return nn::Matrix::Identity(v.rows(), v.cols());
    }
};
}  // namespace

namespace test {
void TestReLuEvaluate() {
    nn::ReLu relu;

    // Scalar
    assert(relu.Evaluate(0.5) == 0.5);
    assert(relu.Evaluate(-0.5) == 0);

    // Eigen
    nn::Vector v{3};
    v << 1, 2, 3;
    assert(relu.Evaluate(v).isApprox(v));
    assert(relu.Evaluate(-v).isZero());

    nn::Matrix x{2, 2};
    x << 1, -1, -1, 1;
    nn::Matrix x_ans{2, 2};
    x_ans << 1, 0, 0, 1;
    assert(relu.Evaluate(x).isApprox(x_ans));

    // STL
    std::vector<nn::Scalar> w{1, 2, 3};
    std::vector<nn::Scalar> w_ans = w;
    relu.Evaluate(w.begin(), w.end());
    assert(IsApproxEqual(w.begin(), w.end(), w_ans.begin()));

    std::deque<nn::Scalar> d{1, -1, 1};
    relu.Evaluate(d.begin(), d.end());
    std::deque<nn::Scalar> d_ans = {1, 0, 1};
    assert(IsApproxEqual(d.begin(), d.end(), d_ans.begin()));
}

void TestReLuDerivative() {
    nn::ReLu relu;

    // Scalar
    assert(relu.EvaluateDerivative(0.5) == 1);
    assert(relu.EvaluateDerivative(-0.5) == 0);
    assert(relu.EvaluateDerivative(0) == 0);

    // Eigen
    nn::Vector v{3};
    v << 1, 2, 3;
    assert(relu.GetDifferential(v).isIdentity());
    assert(relu.GetDifferential(-v).isZero());

    // STL
    std::vector<nn::Scalar> w{1, 2, 3};
    std::vector<nn::Scalar> w_ans{1, 1, 1};
    relu.EvaluateDerivative(w.begin(), w.end());
    assert(IsApproxEqual(w.begin(), w.end(), w_ans.begin()));

    std::deque<nn::Scalar> d{10, -10, 10};
    relu.EvaluateDerivative(d.begin(), d.end());
    std::deque<nn::Scalar> d_ans = {1, 0, 1};
    assert(IsApproxEqual(d.begin(), d.end(), d_ans.begin()));
}

void TestLeakyReLuEvaluate() {
    nn::Scalar slope = 0.1;
    nn::LeakyReLu leaky_relu(slope);

    // Scalar
    assert(leaky_relu.Evaluate(0.5) == 0.5);
    assert(leaky_relu.Evaluate(-0.5) == -0.05);

    // Eigen
    nn::Vector v{3};
    v << 1, 2, 3;
    assert(leaky_relu.Evaluate(v).isApprox(v));

    v << -1, -2, -1;
    nn::Vector ans_v{3};
    ans_v << -0.1, -0.2, -0.1;
    assert(leaky_relu.Evaluate(v).isApprox(ans_v));

    nn::Matrix x{2, 2};
    x << 1, -1, -1, 1;
    nn::Matrix x_ans{2, 2};
    x_ans << 1, -0.1, -0.1, 1;
    assert(leaky_relu.Evaluate(x).isApprox(x_ans));

    // STL
    std::vector<nn::Scalar> w{1, 2, 3};
    std::vector<nn::Scalar> w_ans = w;
    leaky_relu.Evaluate(w.begin(), w.end());
    assert(IsApproxEqual(w.begin(), w.end(), w_ans.begin()));

    std::deque<nn::Scalar> d{1, -1, 1};
    leaky_relu.Evaluate(d.begin(), d.end());
    std::deque<nn::Scalar> d_ans = {1, -0.1, 1};
    assert(IsApproxEqual(d.begin(), d.end(), d_ans.begin()));
}

void TestLeakyReLuDerivative() {
    nn::Scalar slope = 0.1;
    nn::LeakyReLu leaky_relu(slope);

    // Scalar
    assert(leaky_relu.EvaluateDerivative(0.5) == 1);
    assert(leaky_relu.EvaluateDerivative(-0.5) == slope);
    assert(leaky_relu.EvaluateDerivative(0) == slope);

    // Eigen
    nn::Vector v{2};
    v << 1, 2;
    assert(leaky_relu.GetDifferential(v).isIdentity());

    v << -1, -2;
    nn::Matrix ans_v{2, 2};
    ans_v << slope, 0, 0, slope;
    assert(leaky_relu.GetDifferential(v).isApprox(ans_v));

    // STL
    std::vector<nn::Scalar> w{1, 2, 3};
    std::vector<nn::Scalar> w_ans{1, 1, 1};
    leaky_relu.EvaluateDerivative(w.begin(), w.end());
    assert(IsApproxEqual(w.begin(), w.end(), w_ans.begin()));

    std::deque<nn::Scalar> d{10, -10, 10};
    leaky_relu.EvaluateDerivative(d.begin(), d.end());
    std::deque<nn::Scalar> d_ans = {1, slope, 1};
    assert(IsApproxEqual(d.begin(), d.end(), d_ans.begin()));
}

void TestLinearEvaluate() {
    nn::Linear linear;

    // Scalar
    assert(linear.Evaluate(0.5) == 0.5);

    // Eigen
    nn::Vector v{3};
    v << 1, 2, 3;
    assert(linear.Evaluate(v).isApprox(v));

    nn::Matrix x{2, 2};
    x << 1, -1, -1, 1;
    assert(linear.Evaluate(x).isApprox(x));

    // STL
    std::vector<nn::Scalar> w{1, 2, 3};
    std::vector<nn::Scalar> w_ans = w;
    linear.Evaluate(w.begin(), w.end());
    assert(IsApproxEqual(w.begin(), w.end(), w_ans.begin()));
}

void TestLinearDerivative() {
    nn::Linear linear;

    // Scalar
    assert(linear.EvaluateDerivative(0.5) == 1);
    assert(linear.EvaluateDerivative(-0.5) == 1);
    assert(linear.EvaluateDerivative(0) == 1);

    // Eigen
    nn::Vector v{3};
    v << 1, 2, 3;
    assert(linear.GetDifferential(v).isIdentity());

    // STL
    std::vector<nn::Scalar> w{1, 2, 3};
    std::vector<nn::Scalar> w_ans{1, 1, 1};
    linear.EvaluateDerivative(w.begin(), w.end());
    assert(IsApproxEqual(w.begin(), w.end(), w_ans.begin()));

    std::deque<nn::Scalar> d{10, -10, 10};
    linear.EvaluateDerivative(d.begin(), d.end());
    std::deque<nn::Scalar> d_ans = {1, 1, 1};
    assert(IsApproxEqual(d.begin(), d.end(), d_ans.begin()));
}

void TestSigmoidEvaluate() {
    nn::Sigmoid sigmoid;

    // Scalar
    assert(sigmoid.Evaluate(0) == 0.5);
    assert(abs(sigmoid.Evaluate(0.5) - 0.62245) < kEps);

    // Eigen
    nn::Vector v{3};
    v << 1, 2, 3;
    nn::Vector ans_v{3};
    ans_v << 0.73105, 0.88079, 0.95257;
    assert(sigmoid.Evaluate(v).isApprox(ans_v, 1e-3));

    nn::Matrix x{2, 2};
    x << 1, 2, 2, 1;
    nn::Matrix ans_x{2, 2};
    ans_x << 0.73105, 0.88079, 0.88079, 0.73105;
    assert(sigmoid.Evaluate(x).isApprox(ans_x, 1e-3));

    // STL
    std::vector<nn::Scalar> w{1, 2, 3};
    std::vector<nn::Scalar> w_ans{0.73105, 0.88079, 0.95257};
    sigmoid.Evaluate(w.begin(), w.end());
    assert(IsApproxEqual(w.begin(), w.end(), w_ans.begin()));
}

void TestSigmoidDerivative() {
    nn::Sigmoid sigmoid;

    // Scalar
    assert(sigmoid.EvaluateDerivative(0) == 0.25);
    assert(abs(sigmoid.EvaluateDerivative(0.5) - 0.23500) < kEps);

    // Eigen
    nn::Vector v{3};
    v << 1, 2, 3;
    nn::Vector ans_v{3};
    ans_v << 0.19661, 0.10499, 0.04517;
    assert(sigmoid.GetDifferential(v).isApprox(ans_v.asDiagonal().toDenseMatrix(), 1e-3));

    // STL
    std::vector<nn::Scalar> w{1, 2, 3};
    std::vector<nn::Scalar> w_ans{0.19661, 0.10499, 0.04517};
    sigmoid.EvaluateDerivative(w.begin(), w.end());
    assert(IsApproxEqual(w.begin(), w.end(), w_ans.begin()));
}

void TestSoftMaxEvaluate() {
    nn::SoftMax softmax;

    nn::Matrix x{2, 2};
    x << 1, 3, 2, 1;
    nn::Matrix ans_x{2, 2};
    ans_x << 0.26894, 0.88079, 0.73105, 0.11920;
    assert(softmax.Evaluate(x).isApprox(ans_x, 1e-3));
}

void TestSoftMaxDerivative() {
    nn::SoftMax softmax;

    nn::Vector x{2};
    x << 1, 2;
    nn::Matrix ans_x{2, 2};
    ans_x << 0.196611, -0.196609, -0.196609, 0.196615;
    assert(softmax.GetDifferential(x).isApprox(ans_x, 1e-3));
}

void TestNewActivation() {
    nn::ActivationFunction func = ExampleActivation();

    nn::Matrix x{2, 2};
    x << 1, 2, 2, 1;
    assert(func->Evaluate(x).isZero());

    nn::Vector v{2};
    v << 1, 2;
    assert(func->GetDifferential(v).isIdentity());
}

void TestActivationCorrectness() {
    nn::ActivationFunction relu = nn::ReLu();

    nn::Matrix x{2, 2};
    x << 1, -1, -1, 1;
    assert(relu->Evaluate(x).isApprox(relu->Evaluate(x)));

    nn::Vector v{2};
    v << 1, 2;
    assert(relu->GetDifferential(v).isApprox(relu->GetDifferential(v)));
}
}  // namespace test
