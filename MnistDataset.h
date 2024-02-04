#pragma once

#include <filesystem>
#include <vector>

#include "declarations.h"

namespace nn {
namespace mnist {
struct Data {
    std::vector<Vector> X;
    std::vector<Vector> y;
};
Data LoadData(const std::filesystem::path &images_path, const std::filesystem::path &labels_path);
}  // namespace mnist
}  // namespace nn
