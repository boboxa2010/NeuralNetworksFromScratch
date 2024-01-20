#include "utils.h"

#include "iostream"

namespace project {

    namespace {
        constexpr size_t kImageSize = 784;
        constexpr size_t kLabelSize = 10;
    }  // namespace

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

    uint32_t ConvToLittleEndian(uint32_t n) {
        char *bytes = reinterpret_cast<char *>(&n);
        return (static_cast<uint8_t>(bytes[3]) | (static_cast<uint8_t>(bytes[2]) << 8) |
                (static_cast<uint8_t>(bytes[1]) << 16) | (static_cast<uint8_t>(bytes[0]) << 24));
    }

    Vector OneHotEncoding(uint8_t object, size_t number_of_categories) {
        Vector encoded(number_of_categories);
        std::fill(encoded.data(), encoded.data() + encoded.size(), 0.0);
        encoded[object] = 1.0;
        return encoded;
    }
}  // namespace project
