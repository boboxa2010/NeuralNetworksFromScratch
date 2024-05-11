#include "Except.h"

#include <stdexcept>

namespace except {
void React() {
    try {
        throw;
    } catch(const std::invalid_argument& e) {
        // обрабатываешь известные исключения
    } catch(...) {
        // обрабатываешь незивестные исключения
    }
}
}
