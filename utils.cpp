#include "utils.h"

#include "EigenRand/EigenRand"
#include "declarations.h"
#include "iostream"

namespace {
constexpr uint32_t kImageSize = 784;
constexpr uint32_t kLabelSize = 10;

constexpr uint32_t kMnistImageMagicNumber = 2051;
constexpr uint32_t kMnistLabelMagicNumber = 2049;
constexpr uint32_t kMnistImageNumberRows = 28;
constexpr uint32_t kMnistImageNumberColumn = 28;
constexpr uint32_t kMnistNumberOfPixels = 784;

constexpr nn::Scalar kNormalizeCoefficient = 1.0 / 255.0;

uint32_t ConvToLittleEndian(uint32_t n) {
    char *bytes = reinterpret_cast<char *>(&n);
    return (static_cast<uint8_t>(bytes[3]) | (static_cast<uint8_t>(bytes[2]) << 8) |
            (static_cast<uint8_t>(bytes[1]) << 16) | (static_cast<uint8_t>(bytes[0]) << 24));
}

nn::Vector GetNormalizedImage(const std::array<uint8_t, kMnistNumberOfPixels> &image) {
    nn::Vector normalized_image(kMnistNumberOfPixels);
    for (size_t i = 0; i < kMnistNumberOfPixels; ++i) {
        normalized_image[i] = kNormalizeCoefficient * image[i];
    }
    return normalized_image;
}

}  // namespace
namespace nn {
void AsciiRender(const Vector &image, const Vector &label) {
    assert(image.size() == kImageSize && label.size() == kLabelSize);
    for (size_t i = 0; i < kLabelSize; ++i) {
        if (label[i] == 1.0) {
            std::cout << "Label is " << i << '\n';
            break;
        }
    }

    for (size_t i = 0; i < 28; ++i) {
        size_t offset = i * 28;
        for (size_t j = 0; j < 28; ++j) {
            if (image[offset + j] > 0.5) {
                if (image[offset + j] > 0.9) {
                    std::cout << '#';
                } else if (image[offset + j] > 0.7) {
                    std::cout << '*';
                } else {
                    std::cout << '.';
                }
            } else {
                std::cout << ' ';
            }
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

Vector OneHotEncoding(uint8_t object, size_t number_of_categories) {
    Vector encoded(number_of_categories);
    std::fill(encoded.data(), encoded.data() + encoded.size(), 0.0);
    encoded[object] = 1.0;
    return encoded;
}

Matrix GenerateRandNMatrix(size_t rows, size_t columns) {
    static Eigen::Rand::Vmt19937_64 urng{42};
    static Eigen::Rand::NormalGen<float> norm_gen{0, 1};
    return norm_gen.generate<Matrix>(rows, columns, urng);
}

Vector GenerateRandNVector(size_t size) {
    return GenerateRandNMatrix(size, 1);
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

    if (magic_number != kMnistLabelMagicNumber) {
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

    if (magic_number != kMnistImageMagicNumber) {
        throw std::invalid_argument{"Invalid format of input file"};
    }

    if (number_of_rows != kMnistImageNumberRows) {
        throw std::invalid_argument("Invalid width of input file");
    }

    if (number_of_rows != kMnistImageNumberColumn) {
        throw std::invalid_argument("Invalid height of input file");
    }

    std::vector<Vector> images;
    std::array<uint8_t, kMnistImageNumberColumn * kMnistImageNumberRows> image{};
    for (size_t i = 0; i < number_of_images; ++i) {
        file.read(reinterpret_cast<char *>(&image), image.size());
        images.push_back(GetNormalizedImage(image));
    }

    return images;
}
}  // namespace nn
