#pragma once

#include <filesystem>

#include "declarations.h"

namespace nn {
void AsciiRender(const Vector &image, const Vector &label);

Vector OneHotEncoding(uint8_t object, size_t number_of_categories);

Matrix GenerateRandNMatrix(size_t rows, size_t columns);

Vector GenerateRandNVector(size_t size);

std::vector<Vector> ReadLabels(const std::filesystem::path &path);

std::vector<Vector> ReadImages(const std::filesystem::path &path);
}  // namespace nn
