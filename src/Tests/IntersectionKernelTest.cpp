//
//  IntersectionKernelTest.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/18/14.
//
//


#include "armadillo"
#include "gtest/gtest.h"
#include "IntersectionKernelTest.h"

#include "../Kernels/ApproximateKernel.h"
#include "../Kernels/Spline.h"

using namespace arma;
// The fixture for testing class Project1. From google test primer.
class IntersectionKernelTest : public ::testing::Test {
public:
    // You can remove any or all of the following functions if its body
    // is empty.


    mat X;
    colvec beta;
    mat x_test;

    IntersectionKernel intKernel;
    IntersectionKernel_fast intKernel_fast;

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


};


TEST(ApproximateKernel, SplineApproximation){

    int n=1000;

    arma::rowvec x=arma::linspace<rowvec>(0, 5,n);
    arma::rowvec y=sin(x);

    //std::cout<<x<<std::endl;
    //std::cout<<y<<std::endl;

    arma::sort(x);
    Spline spline;

    spline.fitSpline(x, y);

    //std::cout<<spline<<std::endl;

    for (int i=0; i<n; i++) {

        double r=spline.evaluate(x[i]);

        //std::cout<<r<<" "<<i<<std::endl;
        EXPECT_NEAR(r, y[i], 1e-6);
    }

   // spline.fi

}



// Test case must be called the class above
// Also note: use TEST_F instead of TEST to access the test fixture (from google test primer)
TEST_F(IntersectionKernelTest, KernelCalculation) {

    double loopKernelValue=0;
    double kernelFast=0;

    int m=this->X.n_rows;

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

TEST_F(IntersectionKernel_tracking_test, DISABLED_Tracking){



    // Parameters
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    params p;
    p.C=100;
    p.n_O=10;
    p.n_R=10;
    int nRadial=5;
    int nAngular=16;
    int B=10;

    //RawFeatures* features=new RawFeatures(16);
    HistogramFeatures* features=new HistogramFeatures(4,16);
    // RBFKe
    IntersectionKernel_fast* kernel_fast=new IntersectionKernel_fast;
    IntersectionKernel* kernel=new IntersectionKernel;
    //Haar* features=new Haar(2);

    int verbose =0;
    int display =1;
    int m=features->calculateFeatureDimension();
    int K=nRadial*nAngular+1;

    OLaRank_old* olarank=new OLaRank_old(kernel);

    OLaRank_old* olarank_fast=new OLaRank_old(kernel_fast);

    olarank->setParameters(p, B,m,verbose);
    olarank_fast->setParameters(p, B,m,verbose);

    int r_search=30;
    int r_update=60;

    bool useFilter=true;
    bool useObjectness=false;
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    LocationSampler* samplerForUpdate=new LocationSampler(r_update,nRadial,nAngular);
    LocationSampler* samplerForSearch=new LocationSampler(r_search,nRadial,nAngular);


    //Struck t(olarank, features,samplerForSearch, samplerForUpdate,useFilter, display);
    //Struck t_fast(olarank_fast, features,samplerForSearch, samplerForUpdate,useObjectness,useFilter, display);

    Struck t_fast(olarank_fast, features, samplerForSearch, samplerForUpdate,
                   useObjectness, false, useFilter, false, display);


    std::string rootFolder="/Users/Ivan/Files/Data/wu2013/";
    std::string saveFolder="";
    DatasetWu2013* datasetWu2013=new DatasetWu2013;


    //tracker.applyTrackerOnDataset(datasetWu2013, rootFolder,6);

    // use only on the first 'basketball' sequence

    int vidNumber=0;
    int nOfFrames=30;

    t_fast.applyTrackerOnVideoWithinRange(datasetWu2013, rootFolder,saveFolder, 0, 0, nOfFrames);

    Struck t(olarank,features,samplerForSearch,samplerForUpdate,useObjectness,false,useFilter,false,display);

    t.applyTrackerOnVideoWithinRange(datasetWu2013, rootFolder,saveFolder, 0, 0, nOfFrames);
    std::vector<cv::Rect> boxes_fast=t_fast.getBoundingBoxes();
    std::vector<cv::Rect> boxes_slow=t.getBoundingBoxes();


    for (int i=0; i<boxes_slow.size(); i++) {
        cv::Rect b1=boxes_slow[i];
        cv::Rect b2=boxes_fast[i];

        EXPECT_GT((b1&b2).area()*1.0/(std::max(b1.area(),b2.area())),0.8);
    }



}
