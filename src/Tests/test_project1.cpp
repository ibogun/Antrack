#include <iostream>
#include "gtest/gtest.h"

#include "project1.h"

#include "../Kernels/IntersectionKernel.h"
#include "../Kernels/IntersectionKernel_fast.h"
#include "armadillo"

using namespace arma;

// IndependentMethod is a test case - here, we have 2 tests for this 1 test case
TEST(IndependentMethod, ResetsToZero) {
	int i = 3;
	independentMethod(i);
	EXPECT_EQ(0, i);

	i = 12;
	independentMethod(i);
	EXPECT_EQ(0,i);
}

TEST(IndependentMethod, ResetsToZero2) {
	int i = 0;
	independentMethod(i);
	EXPECT_EQ(0, i);
}

// The fixture for testing class Project1. From google test primer.
class IntersectionKernelTest : public ::testing::Test {
public:
	// You can remove any or all of the following functions if its body
	// is empty.

	IntersectionKernelTest() {
		// You can do set-up work for each test here.

        
	}

	virtual ~IntersectionKernelTest() {
		// You can do clean-up work that doesn't throw exceptions here.
	}

	// If the constructor and destructor are not enough for setting up
	// and cleaning up each test, you can define the following methods:
	virtual void SetUp() {
		// Code here will be called immediately after the constructor (right
		// before each test).
        
        // set seed if you want reproducible tests
        //arma_rng::set_seed(1);
        int n=300;
        int m=200;
        
        int nTestCases=500;
        using namespace arma;
        arma::mat X=arma::randu<arma::mat>(m,n);

        
        /*
        x1=[1 2 3];
        x2=[2 4 6; 2 ];
        
        arma::mat X;
        X<<2<<4<<6<<endr
        <<2<<1<<0<<endr;
        mat x_t;
        
        x_t<<1<<2<<3<<endr;
        colvec beta;
        beta<<2<<-2<<endr;
         
         */
        
        colvec beta=randu<colvec>(m);
        
        beta(0)=0;
        beta(0)=-sum(beta);
        
 
        this->X=X;
        this->beta=beta;
        
        mat x_t=randu<mat>(nTestCases,n);
  
        this->x_test=x_t;
	}

	virtual void TearDown() {
		// Code here will be called immediately after each test (right
		// before the destructor).
	}

	// Objects declared here can be used by all tests in the test case for Project1.
	Project1 p;
    
    mat X;
    colvec beta;
    mat x_test;
    
    IntersectionKernel intKernel;
    IntersectionKernel_fast intKernel_fast;
};

// Test case must be called the class above
// Also note: use TEST_F instead of TEST to access the test fixture (from google test primer)
TEST_F(IntersectionKernelTest, KernelCalculation) {
    
    double loopKernelValue=0;
    double kernelFast=0;
    
    int m=this->X.n_rows;
    int n=this->X.n_cols;
    
    int nTestCases=this->x_test.n_rows;
    
    intKernel_fast.preprocessMatrices(this->X, this->beta);
    
    for (int i=0; i<nTestCases; i++) {
        loopKernelValue=0;
        kernelFast=0;
        
        for (int j=0; j<m; j++) {
            loopKernelValue+=this->beta(j)*intKernel.calculate(this->x_test, i, this->X, j);
        }
        
        rowvec x_=this->x_test.row(i);
        kernelFast+=intKernel_fast.predictOne(x_);
        
        
        EXPECT_NEAR(kernelFast, loopKernelValue, 1e-6);
        
    }
    
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// }  // namespace - could surround Project1Test in a namespace