include_directories(${NeuralNetworksFromScratch_SOURCE_DIR}/Tests/inc)
include_directories(${NeuralNetworksFromScratch_SOURCE_DIR}/Library/inc)
add_executable(Tests Except.cpp main.cpp TestActivationFunction.cpp TestLoss.cpp TestMnist.cpp Tests.cpp
        TestLayer.cpp)
target_link_libraries(Tests Library)
