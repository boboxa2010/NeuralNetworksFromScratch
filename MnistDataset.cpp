#include "MnistDataset.h"

#include <fstream>

#include "utils.h"

namespace nn::mnist {
Data LoadData(const std::filesystem::path &images_path, const std::filesystem::path &labels_path) {
    Data dataset;
    dataset.X = ReadImages(images_path);
    dataset.y = ReadLabels(labels_path);
    return dataset;
}
}  // namespace nn::mnist
