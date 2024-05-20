cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS Off)

project(NeuralNetworksFromScratch)

include_directories(eigen)
include_directories(EigenRand)

add_subdirectory(Library)
add_subdirectory(Tests)
