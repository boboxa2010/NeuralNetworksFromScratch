#include "../inc/Except.h"

#include <iostream>
#include <stdexcept>

namespace except {
void React() {
    try {
        throw;
    } catch(const std::invalid_argument& e) {
        std::cout << e.what() << '\n';
    } catch(...) {
        std::cout << "Unknown error" << '\n';
    }
}
}
