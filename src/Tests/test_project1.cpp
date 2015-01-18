#include <iostream>
#include "gtest/gtest.h"





#include "IntersectionKernelTest.h"
//#include "IntersectionKernelTest.cpp"



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// }  // namespace - could surround Project1Test in a namespace