#include "MnistDataset.h"
#include "utils.h"
#include <fstream>
namespace project {
namespace {
enum {
    MNIST_IMAGE_MAGIC_NUMBER = 2051,
    MNIST_LABEL_MAGIC_NUMBER = 2049,
    MNIST_IMAGE_NUMBER_ROWS = 28,
    MNIST_IMAGE_NUMBER_COLUMN = 28,
    MNIST_NUMBER_OF_PIXELS = 784,
};

constexpr NumT kNormalizeCoefficient = 1.0 / 255.0;

Vector GetNormalizedImage(const std::array<uint8_t, MNIST_NUMBER_OF_PIXELS> &image) {
    Vector normalized_image(MNIST_NUMBER_OF_PIXELS);
    for (size_t i = 0; i < MNIST_NUMBER_OF_PIXELS; ++i) {
        normalized_image[i] = kNormalizeCoefficient * image[i];
    }
    return normalized_image;
}

std::vector<Vector> ReadLabels(const std::filesystem::path &path) {
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::invalid_argument{"Cannot open file"};
    }
    uint32_t magic_number;
    uint32_t number_of_labels;

    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char *>(&number_of_labels), sizeof(number_of_labels));

    magic_number = ConvToLittleEndian(magic_number);
    number_of_labels = ConvToLittleEndian(number_of_labels);

    if (magic_number != MNIST_LABEL_MAGIC_NUMBER) {
        throw std::invalid_argument{"Invalid format of input file"};
    }

    std::vector<Vector> labels;
    uint8_t label = 0;
    for (size_t i = 0; i < number_of_labels; ++i) {
        file.read(reinterpret_cast<char *>(&label), sizeof(label));
        labels.push_back(OneHotEncoding(label, 10));
    }
    return labels;
}

std::vector<Vector> ReadImages(const std::filesystem::path &path) {
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::invalid_argument{"Cannot open file"};
    }
    uint32_t magic_number;
    uint32_t number_of_images;
    uint32_t number_of_rows;
    uint32_t number_of_columns;

    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char *>(&number_of_images), sizeof(number_of_images));
    file.read(reinterpret_cast<char *>(&number_of_rows), sizeof(number_of_rows));
    file.read(reinterpret_cast<char *>(&number_of_columns), sizeof(number_of_columns));

    magic_number = ConvToLittleEndian(magic_number);
    number_of_images = ConvToLittleEndian(number_of_images);
    number_of_rows = ConvToLittleEndian(number_of_rows);
    number_of_columns = ConvToLittleEndian(number_of_columns);

    if (magic_number != MNIST_IMAGE_MAGIC_NUMBER) {
        throw std::invalid_argument{"Invalid format of input file"};
    }

    if (number_of_rows != MNIST_IMAGE_NUMBER_ROWS) {
        throw std::invalid_argument("Invalid width of input file");
    }

    if (number_of_rows != MNIST_IMAGE_NUMBER_COLUMN) {
        throw std::invalid_argument("Invalid height of input file");
    }

    std::vector<Vector> images;
    std::array<uint8_t, MNIST_IMAGE_NUMBER_COLUMN * MNIST_IMAGE_NUMBER_ROWS> image{};
    for (size_t i = 0; i < number_of_images; ++i) {
        file.read(reinterpret_cast<char *>(&image), image.size());
        images.push_back(GetNormalizedImage(image));
    }

    return images;
}
}  // namespace

Data MnistDataset::LoadData(const std::filesystem::path &images_path,
                            const std::filesystem::path &labels_path) {
    Data dataset;
    dataset.X = ReadImages(images_path);
    dataset.y = ReadLabels(labels_path);
    return dataset;
}
}  // namespace dl