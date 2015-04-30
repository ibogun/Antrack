//
//  OLaRank_old.h
//  Structured_BING
//
//  Created by Ivan Bogun on 7/17/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef OLARANK_OLD_H_
#define OLARANK_OLD_H_
#include <vector>

#include <tuple>
#include "armadillo"

#include "../Kernels/Kernel.h"
#include <unordered_map>
#include "supportData.h"
struct params {

	double C; // C parameter in SVM notation
	int n_R;  // number of outer iterations
	int n_O;  // number of inner iterations

};


using namespace arma;
using namespace std;


struct Key {
    int frameNumber_1;
    int frameNumber_2;

    Key(int frameNumber_1_,int frameNumber_2_){
        this->frameNumber_1=frameNumber_1_;
        this->frameNumber_2=frameNumber_2_;
    }
};

struct KeyHash {
    std::size_t operator()(const Key& k) const
    {
        return std::hash<int>()(k.frameNumber_1*k.frameNumber_2) ;
    }
};

struct KeyEqual {
    bool operator()(const Key& lhs, const Key& rhs) const
    {
        return lhs.frameNumber_1 == rhs.frameNumber_1 && lhs.frameNumber_2 == rhs.frameNumber_2;
    }
};





class OLaRank_old {

private:

    params parameters;

    // number of classes


    Kernel* svm_kernel;



    // if true,
    int verbose;

    // frameNumber - list of <frameNumber,value> pairs
    unordered_map<Key,arma::mat,KeyHash, KeyEqual> kern;


    friend std::ostream& operator<<(std::ostream&, const OLaRank_old&);

public:
    // size of the pattern

    int B;
    int m;
    vector<supportData*> S;

    std::vector<std::pair<double, double>> velocity;

    // populate this first

    // frameNumber - location unordered map
    unordered_map<int,std::pair<double, double>> locations;

    void clear(){

        this->kern.clear();
        this->velocity.clear();
        this->locations.clear();
    };


    //std::vector<std::pair<double,double>> locations;

    OLaRank_old(Kernel*,int seed=1);
    OLaRank_old(Kernel*,params&, int&, int&,int&);

    void setParameters(params&, int&, int&, int&);

    void initialize(mat&, mat&,const int,int);
    int processAndPredict(mat&,mat&,int);
    void process(mat&,mat&,const int,int);

    double kernel(mat, mat, mat, mat);


    double kernel_fast(mat&,mat&,int,int,mat&,mat&,int,int);
    double calculate_kernel(mat&,int,mat&,int);

    double calculateObjective();
    double loss(const mat, const mat);

    tuple<mat, mat, mat> processNew(mat&, mat&, const int,int);
    int budgetMaintance();

    void smoStep(const int, mat&, mat&);

    void testIfObjectiveIncreases();
    tuple<int, mat, mat> processOld();
    tuple<int, mat, mat> optimize();

    void checkIfConstraintsSatisfied();

    int predict(mat&);

    rowvec predictAll(mat&);
    void learn(mat&,mat&);

    double recomputeGradient(int i,int y);
    void deleteKernelValues(int frameNumber);

    ~OLaRank_old();
};


#endif /* STRUCK_H_ */
