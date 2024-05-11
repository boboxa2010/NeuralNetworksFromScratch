#pragma once

#include "declarations.h"
#include "assert.h"
#include "MnistDataset.h"

namespace nn {
Vector OneHotEncoding(uint8_t object, Index number_of_categories);

Matrix GenerateRandNMatrix(Index rows, Index columns);

Vector GenerateRandNVector(Index size);

uint32_t ConvToLittleEndian(uint32_t n);

Data ShuffleData(const Data& data);

Index ArgMax(const Vector& v);
}  // namespace nn
