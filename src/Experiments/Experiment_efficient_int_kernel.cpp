//
//  Experiment_efficient_int_kernel.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/15/14.
//
//

#include "Experiment_efficient_int_kernel.h"

#include <time.h>


ExperimentEfficientIntersectionKernel::ExperimentEfficientIntersectionKernel(int n, int m, int nTestCases){
    
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


/**
 *  Calculates time for preprocessing and kernel calculation using efficient 
 *  kernel computation
 *
 *  @return pair of value with first being time for preprocessing and second time for
 *  computation itself
 */
std::pair<float, float> ExperimentEfficientIntersectionKernel::calculateTimeFastKernel(){
    
    using namespace arma;
    using namespace std;
    double kernelFast=0;

    clock_t t1,t2;
    std::pair<float, float> result;
    int nTestCases=this->x_test.n_rows;
    
    t1=clock();
    this->kernel_fast.preprocessMatrices(this->X, this->beta);
    t2=clock();
    
    result.first=((float)t2-(float)t1)/CLOCKS_PER_SEC;
    
    t1=clock();
    for (int i=0; i<nTestCases; i++) {
     
        kernelFast=0;
     
        
        rowvec x_=this->x_test.row(i);
        kernelFast+=this->kernel_fast.predictOne(x_);
    
    }
    
    t2=clock();
    
    result.second=((float)t2-(float)t1)/CLOCKS_PER_SEC;
    
    return result;
}

/**
 *  Calculates time to compute all kernel values for intersection kernel.
 *
 *  @return Time for kernel calculation
 */
float ExperimentEfficientIntersectionKernel::calculateTimeRegularKernel(){
    double loopKernelValue=0;
    using namespace std;
    
    int m=this->X.n_rows;
    clock_t t1,t2;
    
    int nTestCases=this->x_test.n_rows;
    
    t1=clock();
    for (int i=0; i<nTestCases; i++) {
        loopKernelValue=0;

        
        for (int j=0; j<m; j++) {
            loopKernelValue+=this->beta(j)*this->kernel_simple.calculate(this->x_test, i, this->X, j);
        }
  
    }
    t2=clock();
    
    float result=((float)t2-(float)t1)/CLOCKS_PER_SEC;
    return result;
}


/**
 *  This function is not producing correct result. Used only for reference
 *
 *  @return Time to compute linear kernel
 */
float ExperimentEfficientIntersectionKernel::calculateTimeLinearKernel(){
    double loopKernelValue=0;
    using namespace std;
    
    int m=this->X.n_rows;
    clock_t t1,t2;
    
    int nTestCases=this->x_test.n_rows;
    
    t1=clock();
    for (int i=0; i<nTestCases; i++) {
        loopKernelValue=0;
 
        loopKernelValue=arma::dot(this->x_test.row(i), this->X.row(0));
        
    }
    t2=clock();
    
    float result=((float)t2-(float)t1)/CLOCKS_PER_SEC;
    return result;
}


/**
 *  Perform experiment which will show how fast is fast kernel compared to regular one
 *
 *  @param outputDir Directory where dat file will be saved to
 */
void ExperimentEfficientIntersectionKernel::performExperiment(std::string outputDir){
 
    int featureDimension=480; // 4 levels of histograms with 16 dimension each
    
    using namespace arma;
    
    rowvec numSupportVectors;
    rowvec numLocationsToEvaluate;
    numSupportVectors<<100<<200<<500<<1000<<endr;
    numLocationsToEvaluate<<1<<200<<500<<1000<<2000<<3000<<5000<<6500<<endr;
    
    mat results_regular(numSupportVectors.size(),numLocationsToEvaluate.size(),fill::zeros);
    mat results_fast(numSupportVectors.size(),numLocationsToEvaluate.size(),fill::zeros);
    mat results_fastPreprocessing(numSupportVectors.size(),numLocationsToEvaluate.size(),fill::zeros);
    
    mat results_linearKernel(numSupportVectors.size(),numLocationsToEvaluate.size(),fill::zeros);
    
    for (int i=0; i<numSupportVectors.size(); i++) {
        for (int j=0; j<numLocationsToEvaluate.size(); j++) {
            
            
            
            ExperimentEfficientIntersectionKernel k(featureDimension,numSupportVectors(i),numLocationsToEvaluate(j));
            
            results_regular(i,j)=k.calculateTimeRegularKernel();
            results_linearKernel(i,j)=k.calculateTimeLinearKernel();
            
            std::pair<double, double> times=k.calculateTimeFastKernel();
            results_fast(i,j)=times.second;
            results_fastPreprocessing(i,j)=times.first;

        }

    }
    
    
    results_regular.save(outputDir+"/regular.mat",arma::raw_ascii);
    results_fast.save(outputDir+"/fast.mat",arma::raw_ascii);
    results_fastPreprocessing.save(outputDir+"/preprocessing_fast.mat",arma::raw_ascii);
    
    results_linearKernel.save(outputDir+"/linear.mat",arma::raw_ascii);
    
    numSupportVectors.save(outputDir+"/numSupportVectors.mat",arma::raw_ascii);
    numLocationsToEvaluate.save(outputDir+"/numLocations.mat",arma::raw_ascii);
    
}


int main(int argc, const char * argv[]) {
    std::string outputDir="/Users/Ivan/Code/Tracking/Antrack/matlab/efficient_kernels/data";
    
    ExperimentEfficientIntersectionKernel::performExperiment(outputDir);
}

