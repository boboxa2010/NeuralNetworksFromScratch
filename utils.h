#pragma once

#include "declarations.h"

namespace nn {
Vector OneHotEncoding(uint8_t object, Index number_of_categories);

Matrix GenerateRandNMatrix(Index rows, Index columns);

Vector GenerateRandNVector(Index size);

uint32_t ConvToLittleEndian(uint32_t n);

inline Index ArgMax(const Vector& v) {
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

}  // namespace nn
