#pragma once

#include <filesystem>
#include <vector>

#include "declarations.h"

namespace project {

struct Data {
    std::vector<Vector> X;
    std::vector<Vector> y;
};

class MnistDataset {
public:
    static Data LoadData(const std::filesystem::path &images_path,
                         const std::filesystem::path &labels_path);
};
}  // namespace dl