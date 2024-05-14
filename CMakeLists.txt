cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS Off)

project(NeuralNetworksFromScratch)

set(EIGEN_DIR "/home/ivan/Desktop/eigen-3.4.0")
include_directories(${EIGEN_DIR})
include_directories(EigenRand)

add_subdirectory(Library)
add_subdirectory(Tests)

