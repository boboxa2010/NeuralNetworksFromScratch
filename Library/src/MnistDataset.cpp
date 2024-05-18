#include "../inc/MnistDataset.h"

#include "../inc/BFile.h"
#include "../inc/utils.h"

namespace {
constexpr uint32_t kMnistImageMagicNumber = 2051;
constexpr uint32_t kMnistLabelMagicNumber = 2049;
constexpr uint32_t kMnistImageNumberRows = 28;
constexpr uint32_t kMnistImageNumberColumn = 28;
constexpr uint32_t kMnistNumberOfPixels = 784;
constexpr uint32_t kMnistNumberOfDigits = 10;

constexpr nn::Scalar kNormalizeCoefficient = 1.0 / 255.0;

nn::Vector GetNormalizedImage(const std::array<uint8_t, kMnistNumberOfPixels> &image) {
    nn::Vector normalized_image{kMnistNumberOfPixels};
    for (nn::Index i = 0; i < kMnistNumberOfPixels; ++i) {
        normalized_image(i) = kNormalizeCoefficient * image[i];
    }
    return normalized_image;
}

uint32_t ReadUint32(std::ifstream *file) {
    uint32_t number;
    file->read(reinterpret_cast<char *>(&number), sizeof(number));
    return number;
}
}  // namespace

namespace nn::mnist {
Labels::Data Labels::Read(std::ifstream *file) {
    return ReadLabels(file);
}

uint32_t Labels::ReadHeader(std::ifstream *file) {
    uint32_t magic_number = ConvToLittleEndian(ReadUint32(file));
    uint32_t number_of_labels = ConvToLittleEndian(ReadUint32(file));

    if (magic_number != kMnistLabelMagicNumber) {
        throw std::invalid_argument{"Invalid format of input file"};
    }
    return number_of_labels;
}

Labels::Data Labels::ReadLabels(std::ifstream *file) {
    uint32_t number_of_labels = ReadHeader(file);

    Data labels(kMnistNumberOfDigits, number_of_labels);

    uint8_t label = 0;
    for (Index i = 0; i < number_of_labels; ++i) {
        file->read(reinterpret_cast<char *>(&label), sizeof(label));
        labels.col(i) = OneHotEncoding(label, kMnistNumberOfDigits);
    }
    return labels;
}

Images::Data Images::Read(std::ifstream *file) {
    return ReadImages(file);
}

Images::Header Images::ReadHeader(std::ifstream *file) {
    uint32_t magic_number = ConvToLittleEndian(ReadUint32(file));
    uint32_t number_of_images = ConvToLittleEndian(ReadUint32(file));
    uint32_t number_of_rows = ConvToLittleEndian(ReadUint32(file));
    uint32_t number_of_columns = ConvToLittleEndian(ReadUint32(file));

    if (magic_number != kMnistImageMagicNumber) {
        throw std::invalid_argument{"Invalid format of input file"};
    }

    if (number_of_rows != kMnistImageNumberRows) {
        throw std::invalid_argument("Invalid width of input file");
    }

    if (number_of_columns != kMnistImageNumberColumn) {
        throw std::invalid_argument("Invalid height of input file");
    }
    return Header{number_of_images, number_of_rows, number_of_columns};
}

Images::Data Images::ReadImages(std::ifstream *file) {
    auto info = ReadHeader(file);

    Data images(info.number_of_columns * info.number_of_rows, info.number_of_images);

    std::array<uint8_t, kMnistImageNumberColumn * kMnistImageNumberRows> image{};
    for (Index i = 0; i < info.number_of_images; ++i) {
        file->read(reinterpret_cast<char *>(&image), image.size());
        images.col(i) = std::move(GetNormalizedImage(image));
    }

    return images;
}

Data LoadData(const std::filesystem::path &images_path, const std::filesystem::path &labels_path) {
    iternal::BFile<Images> images(images_path);
    iternal::BFile<Labels> labels(labels_path);

    Data dataset;
    dataset.X = images.ReadData();
    dataset.y = labels.ReadData();

    return dataset;
}
}  // namespace nn::mnist
