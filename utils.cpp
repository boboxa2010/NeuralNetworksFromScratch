#include "utils.h"

#include <iostream>

#include "EigenRand/EigenRand"
#include "declarations.h"

namespace {
constexpr uint32_t kImageSize = 784;
constexpr uint32_t kLabelSize = 10;
}  // namespace

namespace nn {
Vector OneHotEncoding(uint8_t object, Index number_of_categories) {
    Vector encoded = Vector::Zero(number_of_categories);
    encoded[object] = 1.0;
    return encoded;
}

Matrix GenerateRandNMatrix(Index rows, Index columns) {
    static Eigen::Rand::Vmt19937_64 urng{42};
    static Eigen::Rand::NormalGen<Scalar> norm_gen{0, 1};
    return norm_gen.generate<Matrix>(rows, columns, urng);
}

Vector GenerateRandNVector(Index size) {
    return GenerateRandNMatrix(size, 1);
}

uint32_t ConvToLittleEndian(uint32_t n) {
    char *bytes = reinterpret_cast<char *>(&n);
    return (static_cast<uint8_t>(bytes[3]) | (static_cast<uint8_t>(bytes[2]) << 8) |
            (static_cast<uint8_t>(bytes[1]) << 16) | (static_cast<uint8_t>(bytes[0]) << 24));
}
}  // namespace nn
