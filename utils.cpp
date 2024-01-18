#include "utils.h"

#include "iostream"

void AsciiRender(const Eigen::VectorXd &image, const Eigen::VectorXd &label) {
    for (size_t i = 0; i < 10; ++i) {
        if (label[i] == 1.0) {
            std::cout << "Digit is " << i << '\n';
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