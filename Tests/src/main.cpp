#include "../inc/Except.h"
#include "../inc/Tests.h"

int main() {
    try {
        test::RunAllTests();
    } catch(...) {
        except::React();
    }
}
