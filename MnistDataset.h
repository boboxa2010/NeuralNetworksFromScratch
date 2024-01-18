#pragma once

#include <filesystem>
#include <vector>

#include "Eigen/Eigen"

struct Data {
    std::vector<Eigen::VectorXd> X;
    std::vector<Eigen::VectorXd> y;
};

class MnistDataset {
public:
    static Data LoadData(const std::filesystem::path &images_path,
                         const std::filesystem::path &labels_path);
};
