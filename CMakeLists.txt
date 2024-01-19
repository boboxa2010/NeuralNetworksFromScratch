cmake_minimum_required(VERSION 3.27)

project(NeuralNetworksFromScratch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

add_subdirectory(eigen)
add_subdirectory(EigenRand)

add_executable(NeuralNetworksFromScratch main.cpp
        ActivationFunction.cpp
        MSE.cpp
        Layer.cpp
        LearningRate.cpp
        MnistDataset.cpp
        utils.cpp)
