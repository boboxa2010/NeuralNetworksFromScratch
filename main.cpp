#include "tests/Tests.h"
#include "Except.h"

int main() {
    try {
        test::RunAllTests();
    } catch(...) {
        except::React();
    }
}
