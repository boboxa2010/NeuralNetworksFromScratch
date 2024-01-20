#pragma once

#include "declarations.h"

namespace project {
void AsciiRender(const Vector &image, const Vector &label);
uint32_t ConvToLittleEndian(uint32_t n);
Vector OneHotEncoding(uint8_t object, size_t number_of_categories);
}  // namespace project
