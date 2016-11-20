#include <iostream>
#include "../src/run.h"
#include "../src/defines.h"

int main(int argc, char **argv) {
    std::cout << "Simple least squares custom optimization example:" << std::endl;
    std::cout << "Minimize the equation ||Ax-b||^2 = Minimize Sum (a_i x - b)^2" << std::endl;

    // Initialize gflags
    gflags::ParseCommandLineFlags(&argc, &argv, true);


}
