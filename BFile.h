#pragma once

#include "filesystem"
#include "fstream"

template <typename T>
class BFile {
    using Data = typename T::Data;
public:
    explicit BFile(const std::filesystem::path& file_name) {
        file_.open(file_name, std::ios::binary);
        if (!file_.is_open()) {
            throw std::invalid_argument{"Cannot open file"};
        }
    }

    Data ReadData() {
        return T::Read(&file_);
    }

private:
    std::ifstream file_;
};