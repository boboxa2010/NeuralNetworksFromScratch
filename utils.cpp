#include "utils.h"

#include <random>

#include "EigenRand/EigenRand"

namespace {
constexpr uint8_t kSeed = 42;
}  // namespace

namespace nn {
Vector OneHotEncoding(uint8_t object, Index number_of_categories) {
    Vector encoded = Vector::Zero(number_of_categories);
    encoded[object] = 1.0;
    return encoded;
}

Matrix GenerateRandNMatrix(Index rows, Index columns) {
    static Eigen::Rand::Vmt19937_64 urng{kSeed};
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

Data ShuffleData(const Data &data) {
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(data.X.cols());
    perm.setIdentity();
    std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(),
                 std::mt19937(std::random_device()()));
    return Data{data.X * perm, data.y * perm};
}

Index ArgMax(const Vector &v) {
    return std::distance(v.data(), std::max_element(v.data(), v.data() + v.size()));
}
}  // namespace nn
