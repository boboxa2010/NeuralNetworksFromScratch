# NeuralNetworksFromScratch

NeuralNetworksFromScratch - это header-only библиотека для работы с полносвязнными нейросетями, написанная на C++17.

## Requirements

* Eigen 3.4.0
* EigenRand 0.5.0
* C++17-compatible compilers

## Поддерживаемый функционал
* ```Network``` - класс полносвязной нейросети
* ```Layer``` - класс полносвязного слоя 
* ```ActivationFunction``` - интерфейс функции активации
* Функции активации: ```Sigmoid```, ```ReLu```, ```LeakyReLu```, ```SoftMax```
* ```LossFunction``` - интерфейс функции потерь
* Функции потерь: ```MSE```, ```CrossEntropy```
* ```LearningRate``` - интерфейс обучающего коэффициента
* Поддерживаемые обучающие коэффициенты: ```Constant```, ```Gradual```, ```VowpalWabbit```
* ```LoadData``` - функция для чтения датасета ```MNIST```

## Внешние библиотеки
В данном проекте используется библиотеки [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (матрично-векторые операции) и [EigenRand](https://bab2min.github.io/eigenrand/v0.5.0/en/index.html) (генерация случайных матриц).
Список используемых alias-ов:
```
using Index = Eigen::Index;
using Scalar = double;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Matrix = Eigen::MatrixXd;
using TenZor = std::vector<Matrix>;

struct Data {
Matrix X;
Matrix y;
};
```
### Установка Библиотек
```
git submodule update --init --recursive
```

## Сборка
```sh
# В директории NeuralNetworksFromScratch.
mkdir build
cd build
cmake ..
make
```

## Примеры использования библиотеки
Библиотека находится в ```namespace nn```. Для более детального изучения библиотеки можно ознакомится с [сопроводительной документацией](NeuralNetworksFromScratch/docs.pdf)
### Конструирование нейросети и обучение
```
#include "Network.h"
#include "MnistDataset.h"

int main() {
    nn::Data train =
        nn::mnist::LoadData("../../../data/train-images.idx3-ubyte", "../../../data/train-labels.idx1-ubyte");
    nn::Data test =
        nn::mnist::LoadData("../../../data/t10k-images.idx3-ubyte", "../../../data/t10k-labels.idx1-ubyte");
    
    nn::Network net({784, 32, 90, 10}, {nn::ReLu(), nn::Sigmoid(), nn::SoftMax()});
    nn::LossFunction loss = nn::MSE();
    nn::LearningRate lr = nn::Constant(1);
    net.Train(train, 10, 32, loss, lr, test);
}
```
### Использование кастомных функций

```
#include "ActivationFunctions.h"

class ExampleActivation {
public:
    nn::Matrix Evaluate(const nn::Matrix &v) const {
        return nn::Matrix::Identity(v.rows(), v.cols());
    }
    nn::Matrix GetDifferential(const nn::Vector &v) const {
        return nn::Matrix::Zeros(v.rows(), v.cols());
    }
};

int main() {
    nn::ActivationFunction func = ExampleActivation();
    nn::Matrix x{2, 2};
    x << 1, 2, 2, 1;
    auto func->Evaluate(x);
}
```