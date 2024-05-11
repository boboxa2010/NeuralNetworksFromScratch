cmake_minimum_required(VERSION 3.27)

project(NeuralNetworksFromScratch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

include_directories(eigen)
include_directories(EigenRand)

add_executable(NeuralNetworksFromScratch main.cpp
        ActivationFunctions.cpp
        Layer.cpp
        LossFunctions.cpp
        LearningRate.cpp
        MnistDataset.cpp
        utils.cpp
        Timer.cpp
        Network.cpp
        Except.cpp
        tests/Tests.cpp
        tests/TestActivationFunction.cpp
        tests/TestLoss.cpp
        tests/TestMnist.cpp
)
