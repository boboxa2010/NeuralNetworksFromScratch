#pragma once

#include <filesystem>
#include <fstream>
#include <vector>

#include "declarations.h"

namespace nn {

struct Data {
    Matrix X;
    Matrix y;
};

namespace mnist {
class Labels {
public:
    using Data = Matrix;

    static Data Read(std::ifstream* file);

private:
    static uint32_t ReadHeader(std::ifstream* file);

    static Data ReadLabels(std::ifstream* file);
};

class Images {
public:
    using Data = Matrix;

    static Data Read(std::ifstream* file);

private:
    struct Header {
        uint32_t number_of_images = 0;
        uint32_t number_of_rows = 0;
        uint32_t number_of_columns = 0;
    };
    static Images::Header ReadHeader(std::ifstream* file);

    static Data ReadImages(std::ifstream* file);
};

Data LoadData(const std::filesystem::path& images_path, const std::filesystem::path& labels_path);
}  // namespace mnist
}  // namespace nn
