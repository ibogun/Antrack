//
//  OLaRank_old.cpp
//  Structured_BING
//
//  Created by Ivan Bogun on 7/17/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//



#define ARMA_NO_DEBUG
#include <vector>
#include "OLaRank_old.h"
#include <tuple>
#include "armadillo"
#include  <math.h>
#include <assert.h>
#include <iostream>
#include <algorithm>    // std::random_shuffle      // std::vector
#include <ctime>        // std::time
#include <cstdlib>
#include <cfloat>

using namespace arma;
using namespace std;


OLaRank_old::OLaRank_old(Kernel* svm_kernel_,int seed){
    srand (seed);
    svm_kernel=svm_kernel_;
    //this->kern=new unordered_map<Key,arma::mat,KeyHash, KeyEqual>;
}

/***
 *  Set parameters. Used to replace constructor with parameters.
 *
 * @param learningParams
 * @param balance
 * @param m_
 * @param K_
 * @param verbose_
 */
void OLaRank_old::setParameters(params& learningParams, int& balance, int& m_,int& verbose_) {


    parameters = learningParams;

    m = m_;
    B = balance;
    verbose = verbose_;
}


rowvec OLaRank_old::predictAll(mat& newX){


    return this->svm_kernel->predictAll(newX, this->S,this->B);

}


/**
 * The function checks if constraints in multi-class SVM are satisfied
 */
void OLaRank_old::checkIfConstraintsSatisfied(){
    double sum=0;
    for (int i = 0; i < this->S.size(); ++i) {

        sum=0;

        for (int y = 0; y < this->S[i]->x->n_rows; ++y) {

            assert((*this->S[i]->beta)(y)<=this->parameters.C*(this->S[i]->label==y));
            sum+=(*this->S[i]->beta)(y);
        }

        if (abs(sum)>0.00001){
            cout<<"Pattern: "<<i<<" out of "<<this->S.size()<<"\n";
            cout<<this->S[i]->beta;
            cout<<abs(sum)<<" \n";
        }

        assert(abs(sum)<0.00001);

    }
}

/**
 * The function calculates gradient for newly added pattern and a label; it also
 * produces (y_plus,y_neg, gradient)
 *
 * @param newX			- 		pattern
 * @param y_hat 		- 		label ( set of possible positions)
 * @param label 		- 		predicted position
 * @return 				- 		triplet (y_plus,y_neg,gradient)
 */
tuple<mat, mat, mat> OLaRank_old::processNew(mat& newX, mat& y_hat,const int label,int frameNumber) {
    // calculate gradient for the new example


    mat y_bar_mat;

    int K=newX.n_rows;

    mat grad(1, K, fill::zeros);

    for (int y = 0; y < K; ++y) {

        grad(y) -= this->loss(y_hat.row(label), y_hat.row(y));



        for (int i = 0; i < this->S.size(); ++i) {
            for (int y_bar = 0; y_bar < this->S[i]->x->n_rows; ++y_bar) {
                supportData* S_i=this->S[i];
                if ((*S_i->beta)(y_bar)!=0){

                    grad(y) -= (*this->S[i]->beta)(y_bar)
                    * this->kernel_fast(*this->S[i]->x,*this->S[i]->y, (*this->S[i]->y).row(y_bar)(0),(*this->S[i]).frameNumber, newX,y_hat, y_hat.row(y)(0),frameNumber);

                    //					grad(y) -= this->S[i].beta(y_bar)
                    //                    * this->kernel(this->S[i].x, this->S[i].y.row(y_bar), newX, y_hat.row(y));
                }
            }

        }
    }

    uword y_neg_idx;

    grad.min(y_neg_idx);

    //  mat y_plus = y_hat.row(label);
    //	mat y_neg=y_hat.row(y_neg_idx);

    if (this->verbose > 1) {
        cout << "ProcessNew step: \n";
        cout << "(y_plus,y_neg) " << y_hat.row(label)(0) << " " << y_hat.row(y_neg_idx)(0) << endl;
        cout << "--------------------" << endl;
    }

    //tuple<mat, mat, mat> result = make_tuple(y_plus, y_neg, grad);
    tuple<mat, mat, mat> result = make_tuple(y_hat.row(label), y_hat.row(y_neg_idx), grad);
    return result;

}



/**
 * Function which maintains that the number of support patterns is less or equal
 * to the balance (variable B).
 */
int OLaRank_old::budgetMaintance() {
    //int n=this->S.size();

    int test=0;

    vector<int> idxToBeDeleted;

    for (int i=0; i<this->S.size(); ++i) {

        int sum=0;

        for (int j=0; j< this->S[i]->x->n_rows; ++j) {
            //if (abs((*this->S[i]->beta)(j))>=1.0e-7) {

            if(abs((*this->S[i]->beta)(j))!=0){
                test+=1;
            }else{
                sum++;
            }
        }


        //cout<<"Idx: "<<i<<" zeros: "<<sum<<endl;

        if (sum==this->S[i]->x->n_rows) {

            //std::cout<<"Something will be deleted"<<endl;
            idxToBeDeleted.push_back(i);


            //cout<<"Something should be deleted"<<endl;
        }
    }
    //std::cout<<"Before. Number of  support Vectors : "<<test<<" out of: "<<this->S.size()<<std::endl;



    // if the list is larger -> sort it first and delete starting from the highest
    if (idxToBeDeleted.size()>1) {
        std::sort(idxToBeDeleted.begin(), idxToBeDeleted.end());
    }

    // be
    auto beginningIterator=this->S.begin();
    for (int i=(int)idxToBeDeleted.size()-1; i>=0; i--) {
        //cout<<"deleting idx: "<<idxToBeDeleted[i]<<endl;

        // delete kernel values
        deleteKernelValues(this->S[idxToBeDeleted[i]]->frameNumber);

        delete this->S[idxToBeDeleted[i]];
        this->S.erase(beginningIterator+idxToBeDeleted[i]);
    }

    //std::cout<<"Number of  support Vectors : "<<test<<" out of: "<<this->S.size()<<std::endl;
    int n=test;

    //	// First, if there are rows whose betas are all zero - delete them.
    //	for (int i = 0; i < this->S.size(); ++i) {
    //
    //		double absSum=sum(sum(abs(this->S[i].beta)));
    //		if (absSum<1e-20){
    //            //std::cout<<"Deleting the whole row "<<max(abs(this->S[i].beta))<<std::endl;
    //
    //			this->S.erase(this->S.begin()+i);
    //
    //           			return i;
    //		}
    //	}


    // If balance is not exceeded - return.
    if (n<=this->B){
        return -1;
    }

    int maxNRows=0;

    for (int i=0; i<this->S.size(); i++) {
        if (this->S[i]->x->n_rows>=maxNRows) {
            maxNRows=this->S[i]->x->n_rows;
        }
    }

    mat scores((int)this->S.size(),maxNRows,fill::ones);
    mat y_r,x_r;
    scores=scores*INFINITY;

    for (int i = 0; i < this->S.size(); ++i) {
        for (int k = 0; k < this->S[i]->x->n_rows; ++k) {
            if ((*this->S[i]->beta)(k)<0){

                x_r=*this->S[i]->x;
                y_r=*this->S[i]->y;
                //				scores(i,k)=(pow(this->S[i].beta(k),2))*(this->kernel(x_r,y_r.row(k),
                //                                                                      x_r,y_r.row(k))+(this->kernel(x_r,y_r.row(this->S[i].label),
                //                                                                                                    x_r,y_r.row(this->S[i].label)))-
                //                                                         2*this->kernel(x_r,y_r.row(this->S[i].label),
                //                                                                        x_r,y_r.row(k)));


                scores(i,k)=(pow((*this->S[i]->beta)(k),2))*(this->kernel_fast(*this->S[i]->x,y_r,y_r.row(k)(0),(*this->S[i]).frameNumber,
                                                                               *this->S[i]->x,y_r,y_r.row(k)(0),(*this->S[i]).frameNumber)+
                                                             (this->kernel_fast(*this->S[i]->x,y_r,y_r.row(this->S[i]->label)(0),(*this->S[i]).frameNumber,*this->S[i]->x,y_r, y_r.row(this->S[i]->label)(0),(*this->S[i]).frameNumber))-
                                                             2*this->kernel_fast(*this->S[i]->x,y_r,y_r.row(this->S[i]->label)(0),(*this->S[i]).frameNumber,*this->S[i]->x,y_r,y_r.row(k)(0),(*this->S[i]).frameNumber));
            }
        }
    }


    uword I,J;
    scores.min(I,J);

    int numberOfNegativeSP=0;

    vector<int> negativeSP;
    vector<int> positiveSP;

    for (int i = 0; i < this->S[I]->x->n_rows; ++i) {
        if ((*this->S[I]->beta)(i)<0){
            numberOfNegativeSP++;
            negativeSP.push_back(i);
        }

        if ((*this->S[I]->beta)(i)>0){
            positiveSP.push_back(i);
        }
    }


    if (this->verbose>=1){

        cout<<"Balancing step. Indices: I="<<I<<" J=="<<J<<"\n";
        cout<<"Max value "<<scores(I,J)<<"\n";

    }





    if (numberOfNegativeSP==1){
        // if there is only one negative support vector - delete the whole pattern
        // there might be a number of positive vectors. Adjust gradients before deleting the row.

        std::vector<int> allSPtobeDeleted;

        allSPtobeDeleted.reserve( positiveSP.size() + negativeSP.size() ); // preallocate memory
        allSPtobeDeleted.insert( allSPtobeDeleted.end(), positiveSP.begin(), positiveSP.end() );
        allSPtobeDeleted.insert( allSPtobeDeleted.end(), negativeSP.begin(), negativeSP.end() );





        for (int z=0; z<allSPtobeDeleted.size(); ++z) {
            // delete beta(I,z)
            int idx=allSPtobeDeleted[z];
            for (int ii=0; ii<this->S.size(); ++ii) {
                for (int yy=0; yy<this->S[ii]->x->n_rows; ++yy) {
                    (*this->S[ii]->grad)(yy)+=(*this->S[I]->beta)(idx)*kernel_fast(*this->S[ii]->x,*this->S[ii]->y, (*this->S[ii]->y).row(yy)(0),(*this->S[ii]).frameNumber,*this->S[I]->x,*this->S[I]->y, (*this->S[I]->y).row(idx)(0),(*this->S[I]).frameNumber);
                }
            }
        }

        deleteKernelValues(this->S[I]->frameNumber);
        // Problem here;
        delete this->S[I];
        this->S.erase(this->S.begin()+(I));


        // debug code - to see if the leak can be dealt with
        //        std::vector<supportData> newS;
        //        for (int i=0; i<this->S.size(); i++) {
        //            if (i!=I) {
        //                supportData tmp=this->S[i];
        //                newS.push_back( tmp );
        //            }
        //
        //        }
        //
        //        this->S.clear();
        //        this->S=newS;

        //        for (int ii=0; ii<this->S.size(); ++ii) {
        //            for (int yy=0; yy<this->K; ++yy) {
        //                double grad=recomputeGradient(ii, yy);
        //
        //                if(abs(grad-this->S[ii].grad(yy)>0.0000001)){
        //
        //                    std::cout<<"Gradient is not correct in IF expected: "<<grad<<" got: "<<this->S[ii].grad(yy)<<std::endl;
        //                }
        //            }
        //        }


        return I;
    }else{
        // delete negative support vector; in order to compensate (satisfy sum(beta(I,:))=0)
        // add it to the positive support vector
        if ((*this->S[I]->beta)(this->S[I]->label)<0){
            std::cout<<"Should be positive support pattern is not positive"<<std::endl;
        }


        std::vector<int> allSPtobeDeleted;
        allSPtobeDeleted.push_back(J);
        //for (int z=0; z<allSPtobeDeleted.size(); ++z) {
        // delete beta(I,z)
        int idx=J;
        for (int ii=0; ii<this->S.size(); ++ii) {
            for (int yy=0; yy<this->S[ii]->x->n_rows; ++yy) {
                (*this->S[ii]->grad)(yy)+=(*this->S[I]->beta)(idx)*kernel_fast(*this->S[ii]->x,*this->S[ii]->y, (*this->S[ii]->y).row(yy)(0),(*this->S[ii]).frameNumber,*this->S[I]->x,*this->S[I]->y, (*this->S[I]->y).row(idx)(0),(*this->S[I]).frameNumber);
            }
        }

        idx=this->S[I]->label;
        for (int ii=0; ii<this->S.size(); ++ii) {
            for (int yy=0; yy<this->S[ii]->x->n_rows; ++yy) {
                (*this->S[ii]->grad)(yy)-=(*this->S[I]->beta)(J)*kernel_fast(*this->S[ii]->x,(*this->S[ii]->y), (*this->S[ii]->y).row(yy)(0),(*this->S[ii]).frameNumber,*this->S[I]->x,*this->S[I]->y, (*this->S[I]->y).row(idx)(0),(*this->S[I]).frameNumber);
            }
        }

        //std::cout<<"this changes everything"<<std::endl;

        (*this->S[I]->beta)(this->S[I]->label)+=(*this->S[I]->beta)(J);

        (*this->S[I]->beta)(J)=0;


        //        for (int ii=0; ii<this->S.size(); ++ii) {
        //            for (int yy=0; yy<this->K; ++yy) {
        //                double grad=recomputeGradient(ii, yy);
        //
        //                if(abs(grad-this->S[ii].grad(yy)>0.0000001)){
        //
        //                    std::cout<<"Gradient is not correct in ELSE expected: "<<grad<<" got: "<<this->S[ii].grad(yy)<<std::endl;
        //                }
        //            }
        //        }


    }

    if (n-1>this->B) {

        //cout<<"This should never happen but"<<endl;
        budgetMaintance();
    }


    return -1;

}

/**
 * SMO type step for Struck algorithm. All gradients will be updated in this
 * function, as	 well as beta for the i pattern.
 *
 * @param i  		- index of the pattern whose beta will be updated
 * @param y_plus 	- y plus
 * @param y_neg 	- y minus
 */
void OLaRank_old::smoStep(const int i, mat& y_plus, mat& y_neg) {



    double k_00=this->kernel_fast(*this->S[i]->x,*this->S[i]->y, y_plus(0),(*this->S[i]).frameNumber, *this->S[i]->x,*this->S[i]->y,y_plus(0),(*this->S[i]).frameNumber);
    double k_11=this->kernel_fast(*this->S[i]->x,*this->S[i]->y, y_neg(0),(*this->S[i]).frameNumber, *this->S[i]->x,*this->S[i]->y,y_neg(0),(*this->S[i]).frameNumber);
    double k_01=this->kernel_fast(*this->S[i]->x,*this->S[i]->y, y_plus(0),(*this->S[i]).frameNumber, *this->S[i]->x,*this->S[i]->y,y_neg(0),(*this->S[i]).frameNumber);


    double lambda_u = ((*this->S[i]->grad)(y_plus(0)) - (*this->S[i]->grad)(y_neg(0)))
    / (k_00 + k_11 - 2 * k_01);

    double tmp2 = this->parameters.C * (int)(y_plus(0) == this->S[i]->label)
    - (*this->S[i]->beta)(y_plus(0));
    double const tmp = min(lambda_u, tmp2);

    double lambda = max(double(0), tmp);



    (*this->S[i]->beta)(y_plus(0)) += lambda;
    (*this->S[i]->beta)(y_neg(0)) -= lambda;

    double k_0, k_1;

    if (lambda!=0){
        for (int j = 0; j < this->S.size(); ++j) {
            //for (int y = 0; y < this->K; ++y) {
            for (int y = 0; y < this->S[j]->x->n_rows; ++y) {

                k_0=this->kernel_fast(*this->S[j]->x,*this->S[j]->y, (*this->S[j]->y).row(y)(0),(*this->S[j]).frameNumber, *this->S[i]->x, *this->S[i]->x,y_plus(0),(*this->S[i]).frameNumber);
                k_1=this->kernel_fast(*this->S[j]->x,*this->S[j]->y,  (*this->S[j]->y).row(y)(0),(*this->S[j]).frameNumber, *this->S[i]->x, *this->S[i]->y,y_neg(0),(*this->S[i]).frameNumber);


                (*this->S[j]->grad)(y) -= lambda * (k_0 - k_1);

                //}
            }
        }
    }

    if (this->verbose > 1) {
        cout << "SMO step: lambda=" << lambda << '\n';
        cout << "--------------------" << endl;
    }

}


/**
 * Process old
 *
 * @return triplet for SMOstep
 */
tuple<int, mat, mat> OLaRank_old::processOld() {



    int i = rand() % this->S.size();

    mat y_plus,y_neg;
    double grad_max = -INFINITY;
    //cout<<"MAX VALUE: "<<grad_max;
    uword y_neg_idx;

    for (int y = 0; y < this->S[i]->x->n_rows; ++y) {

        if ((*this->S[i]->beta)(y) < (y == this->S[i]->label) * this->parameters.C) {

            if ((*this->S[i]->grad)(y) > grad_max) {
                grad_max = (*this->S[i]->grad)(y);
                y_plus = (*this->S[i]->y).row(y);
            }

        }

    }

    (*this->S[i]->grad).min(y_neg_idx);
    y_neg=(*this->S[i]->y).row(y_neg_idx);

    if (this->verbose > 1) {
        cout << "ProcessOld step: \n" << "(i,y_plus,y_neg) " << i << " "
        << y_plus(0) << " " << y_neg(0) << " " << endl;
        cout << "--------------------" << endl;
    }

    tuple<int, mat, mat> result = make_tuple(i, y_plus, y_neg);
    return result;
}


/**
 * Optimize
 *
 * @return triplet for SMOstep
 */
tuple<int, mat, mat> OLaRank_old::optimize() {

    int i = rand() % this->S.size();

    while (sum(sum(abs(*this->S[i]->beta)))==0) {
        i = rand() % this->S.size();
    }

    mat y_plus,y_neg;
    double grad_max = -INFINITY;
    double grad_min= INFINITY;
    //cout<<"MAX VALUE: "<<grad_max;
    //cout<<this->S[i].beta;
    for (int y = 0; y < this->S[i]->x->n_rows; ++y) {
        //
        //		if (this->S[i].beta(y) != 0
        //            && this->S[i].beta(y)
        //            < this->parameters.C * (y == this->S[i].label)) {
        //
        //			if (this->S[i].grad(y) > grad_max) {
        //				grad_max = this->S[i].grad(y);
        //				y_plus = this->S[i].y.row(y);
        //			}
        //
        //			if (this->S[i].grad(y) < grad_min) {
        //				grad_min = this->S[i].grad(y);
        //				y_neg = this->S[i].y.row(y);
        //			}
        //
        //
        //		}



        if ((*this->S[i]->beta)(y)
            < this->parameters.C * (int)(y == this->S[i]->label)) {

            if ((*this->S[i]->grad)(y) > grad_max) {
                grad_max = (*this->S[i]->grad)(y);
                y_plus = (*this->S[i]->y).row(y);
            }




        }



        if ((*this->S[i]->beta)(y)!=0){
            if ((*this->S[i]->grad)(y) < grad_min) {
                grad_min = (*this->S[i]->grad)(y);
                y_neg = (*this->S[i]->y).row(y);
            }
        }


    }

    if (this->verbose >= 1) {
        //cout<<this->S[i].beta<<endl;
        cout << "Optimize step: \n" << "(i,y_plus,y_neg) " << i << " " << y_plus(0)
        << " " << y_neg(0) << " " << endl;
        cout << "--------------------" << endl;
    }

    tuple<int, mat, mat> result = make_tuple(i, y_plus, y_neg);
    return result;
}

/**
 * Predict a label for the new pattern
 *
 * @param newX - pattern to be  used for prediction
 * @return label of the pattern
 */
int OLaRank_old::predict(mat& newX) {

    //vec F(this->K, fill::zeros);

    rowvec y_hat(1,fill::zeros);
    rowvec y(1,fill::zeros);

    double current=-INFINITY;
    double best=current;
    int bestIdx=0;

    for (int k = 0; k < newX.n_rows; ++k) {

        y(0)=k;
        current=0;


        for (int i = 0; i < this->S.size(); ++i) {
            for (int yhat = 0; yhat < this->S[i]->x->n_rows; ++yhat) {

                y_hat(0)=yhat;

                if ((*this->S[i]->beta)(yhat)!=0){

                    // the below has to be multiplied by the velocities kernel
                    current+= (*this->S[i]->beta)(yhat)
                    * this->calculate_kernel(newX, y(0), *this->S[i]->x, y_hat(0));

                }
            }

        }

        if (current>=best) {
            bestIdx=k;
            best=current;
        }

    }

    //	uword idx;
    //
    //	F.max(idx);

    //    if (idx!=bestIdx){
    //        std::cout<<" NOT RIGHT"<<std::endl;
    //    }

    return bestIdx;
}

/**
 *  Loss function
 * @param y
 * @param y_hat
 * @return value of the loss function
 */
double OLaRank_old::loss(const mat y, const mat y_hat) {

    // find intersection of the two rectangles


    double a1=y(1);
    double b1=y(2);
    double a2=a1+y(3);
    double b2=b1+y(4);

    double c1=y_hat(1);
    double d1=y_hat(2);
    double c2=c1+y_hat(3);
    double d2=d1+y_hat(4);

    double intersection=std::max(std::min(a2,c2)-std::max(a1,c1),double(0))*(std::max(std::min(b2,d2)-std::max(b1,d1),double(0)));


    double loss=1-intersection/(y(3)*y(4)+y_hat(3)*y_hat(4)-intersection);

    return loss;
}

/**
 * Calculates objective function for structured output SVM.
 * @return objective value
 */
double OLaRank_old::calculateObjective() {

    double objective = 0;
    double s = 0;
    for (int i = 0; i < this->S.size(); ++i) {
        for (int k = 0; k < this->S[i]->x->n_rows; ++k) {
            objective += loss((*this->S[i]->y).row(this->S[i]->label), (*this->S[i]->y).row(k)) * (*this->S[i]->beta)(k);
        }

        for (int j = i; j < this->S.size(); ++j) {

            for (int y = 0; y <  this->S[i]->x->n_rows; ++y) {
                for (int yhat = 0; yhat <  this->S[j]->x->n_rows; ++yhat) {

                    s = ((*S[i]->beta)(y) * (*S[j]->beta)(yhat)
                         * kernel(*S[i]->x, (*S[i]->y).row(y), *S[j]->x, (*S[i]->y).row(yhat)));

                    if (i == j) {
                        s = s * 0.5;
                    }

                    objective -= s;
                }
            }
        }
    }

    return objective;
}

/**
 * Calculate kernel value based on the intput patterns/labels
 *
 * @param x pattern 1
 * @param y	label of the pattern 1
 * @param xp pattern 2
 * @param yp label of the pattern 2
 * @return K(x,y,xp,yp);
 */
double OLaRank_old::kernel(mat x, mat y, mat xp, mat yp) {


    double result = exp(-0.2*norm(x.row(y(0))-xp.row(yp(0))));
    return result;
}

double OLaRank_old::kernel_fast(mat& x,mat& y_loc, int y,int frameNumber_1, mat& xp,mat& yp_loc, int yp,int frameNumber_2){


    // to save only half of the kernels
    if (frameNumber_2<frameNumber_1) {
        return kernel_fast(xp,yp_loc, yp, frameNumber_2, x,y_loc, y, frameNumber_1);
    }

    // we can always assume that frameNumber_1>=frameNumber_2
    Key key(frameNumber_1,frameNumber_2);



    auto it=(this->kern).find(key);


    double result=0;

    if (it==(this->kern).end()) {
        // key is not found

        // allocate memory for the new matrix

        arma::mat kern_matrix(x.n_rows,xp.n_rows,arma::fill::ones);

        // set all elements to -infinity which means that the kernel value wasn't calculated
        kern_matrix=(DBL_MIN)*(kern_matrix);

        // calculate the value

        result=calculate_kernel(x, y, xp, yp);

        (kern_matrix)(y,yp)=result;
        // add it to the kernel map
        (this->kern).insert({key,kern_matrix});



    }else{
        // key is found - check if value is calculated

        if ((it->second)(y,yp)!=DBL_MIN) {
            // the value was previously calculated - return it
            result=(it->second)(y,yp);

        }else{
            // the value wasn't calculated. Firstly, calculate it and store in the kernel

            result=calculate_kernel(x, y, xp, yp);
            (it->second)(y,yp)=result;
            // check if it will change
        }

    }


    return result;

}

double OLaRank_old::calculate_kernel(mat& x, int y, mat& xp, int yp){

    return this->svm_kernel->calculate(x,y,xp,yp);
}


/**
 *  Create an instance using given set of parameters
 */
OLaRank_old::OLaRank_old(Kernel* svm_kernel_,params& learningParams, int& balance, int& m_,
                         int& verbose_) {

    svm_kernel=svm_kernel_;
    parameters = learningParams;

    m = m_;
    B = balance;
    verbose = verbose_;

    //this->kern=new unordered_map<Key,arma::mat*,KeyHash, KeyEqual>;
}

/**
 * Predict label for the new pattern, newX, and process it.
 *
 * @param newX 				- 		pattern
 * @return 					- 		predicted label for the pattern
 */
int OLaRank_old::processAndPredict(mat& newX, mat& newY,int frameNumber) {

    int y_hat_idx = this->predict(newX);

    //mat y_hat=newY.row(y_hat_idx);

    tuple<mat, mat, mat> p_new = this->processNew(newX, newY,y_hat_idx,frameNumber);

    mat y_plus, y_neg;
    mat grad(1, newX.n_rows, fill::zeros);
    tie(y_plus, y_neg, grad) = p_new;

    // add new element into set S
    supportData* support=new supportData(newX, newY,y_hat_idx, newX.size(), newX.n_rows,frameNumber);
    (*support->grad)= grad;

    double i = this->S.size();
    this->S.push_back(support);


    //    for (int ii=0; ii<this->S.size(); ++ii) {
    //        for (int yy=0; yy<this->K; ++yy) {
    //            double grad=recomputeGradient(ii, yy);
    //
    //            if(abs(grad-this->S[ii].grad(yy)>0.0000001)){
    //
    //                std::cout<<"Gradient is not correct expected: "<<grad<<" got: "<<this->S[ii].grad(yy)<<std::endl;
    //            }
    //        }
    //    }


    smoStep(i, y_plus, y_neg);
    //this->checkIfConstraintsSatisfied();

    budgetMaintance();

    for (int ii = 0; ii < this->parameters.n_R; ++ii) {


        if (this->S.size()!=0){



            tuple<double, mat, mat> p_old = this->processOld();
            tie(i, y_plus, y_neg) = p_old;

            smoStep(i, y_plus, y_neg);

            //this->checkIfConstraintsSatisfied();

            budgetMaintance();


        }

        for (int j = 0; j < this->parameters.n_O; ++j) {
            if (this->S.size()!=0){
                tuple<double, mat, mat> optimize = this->optimize();

                tie(i, y_plus, y_neg) = optimize;
                smoStep(i, y_plus, y_neg);
            }
            //this->checkIfConstraintsSatisfied();

        }

    }

    if (this->verbose > 2) {

        cout << "---------------------------------------\n";
        cout << "OptValue " << this->calculateObjective() << "\n";
        cout << "---------------------------------------\n";
    }

    return y_hat_idx;
}

/**
 * Process (input,output) pair
 *
 * @param newX 					- pattern
 * @param y_hat					- label
 */
void OLaRank_old::process(mat& newX, mat& y_hat, int y_hat_label,int frameNumber) {



    tuple<mat, mat, mat> p_new = this->processNew(newX, y_hat,y_hat_label,frameNumber);

    mat y_plus, y_neg;
    mat grad(1, newX.n_rows, fill::zeros);
    tie(y_plus, y_neg, grad) = p_new;

    // add new element into set S
    supportData* support=new supportData(newX, y_hat,y_hat_label, newX.size(), newX.n_rows,frameNumber);
    (*support->grad) = grad;


    //delete support;

    double i = this->S.size();
    this->S.push_back(support);



    /*

     debug code

     */
    //    int test=0;
    //
    //    for (int i=0; i<this->S.size(); ++i) {
    //
    //        int sum=0;
    //
    //        for (int j=0; j< this->K; ++j) {
    //            //if (abs((*this->S[i]->beta)(j))>=1.0e-7) {
    //
    //            if(abs((*this->S[i]->beta)(j))!=0){
    //                test+=1;
    //            }else{
    //                sum++;
    //            }
    //        }
    //    }
    //
    //    std::cout<<"Number of  support Vectors : "<<test<<"/"<<this->S.size()<<std::endl;
    //
    //
    //    std::vector<Key> keys;
    //    keys.reserve(this->kern->size());
    //
    //
    //    for(auto kv : *this->kern) {
    //        keys.push_back(kv.first);
    //    }
    //
    //    std::cout<<"Cache size: "<<keys.size()<<"/" <<(this->S.size()*(this->S.size()+1))/2<<std::endl;


    smoStep(i, y_plus, y_neg);
    //this->checkIfConstraintsSatisfied();

    budgetMaintance();

    for (int ii = 0; ii < this->parameters.n_R; ++ii) {

        tuple<double, mat, mat> p_old = this->processOld();
        tie(i, y_plus, y_neg) = p_old;

        smoStep(i, y_plus, y_neg);

        //this->checkIfConstraintsSatisfied();

        budgetMaintance();

        for (int j = 0; j < this->parameters.n_O; ++j) {
            tuple<double, mat, mat> optimize = this->optimize();

            tie(i, y_plus, y_neg) = optimize;
            smoStep(i, y_plus, y_neg);

            //this->checkIfConstraintsSatisfied();
        }

    }

    if (this->verbose > 0) {

        cout << "---------------------------------------\n";
        cout << "OptValue " << this->calculateObjective() << "\n";
        cout << "---------------------------------------\n";
    }
}



void OLaRank_old::testIfObjectiveIncreases() {

    int i;
    mat y_plus, y_neg;

    for (int ii = 0; ii < this->parameters.n_R; ++ii) {
        for (int j = 0; j < 1; ++j) {
            tuple<double, mat, mat> optimize = this->optimize();

            tie(i, y_plus, y_neg) = optimize;

            cout<<"Beta (should be non zero): "<<this->S[i]->beta<<"\n";
            cout<< "Label for the current pattern: "<<this->S[i]->label<<"\n";
            smoStep(i, y_plus, y_neg);
            cout << "OptValueCHECK " << this->calculateObjective() << "\n";
        }

    }


}

/**
 *  Initialize the first training pattern and label
 *
 *  @param x  			first pattern
 *  @param label  		label of the first pattern
 *  @param y 			first label
 */
void OLaRank_old::initialize(mat& x, mat& y,const int label,int frameNumber) {

    supportData* s1=new supportData (x, y,label, m, x.n_rows,frameNumber);
    S.push_back(s1);

    this->process(x, y, label, frameNumber);

}

tuple<mat, vec> generateInput(const int& n, const int& m, const double step) {

    mat x(n, m, fill::zeros);
    vec y(n, fill::ones);

    // set seed
    srand(1);

    //centers of three classes - [-1 1], [1 1], [-1 -1]

    x.submat(0, 0, floor(n / 3) - 1, 0) = mat(floor(n / 3), 1, fill::ones)
    * (-1);
    x.submat(0, 1, floor(n / 3) - 1, 1) = mat(floor(n / 3), 1, fill::ones);

    x.submat(floor(n / 3), 0, 2 * floor(n / 3) - 1, 0) = mat(floor(n / 3), 1,
                                                             fill::ones);
    x.submat(floor(n / 3), 1, 2 * floor(n / 3) - 1, 1) = mat(floor(n / 3), 1,
                                                             fill::ones);

    y.subvec(floor(n / 3), 2 * floor(n / 3) - 1) = vec(floor(n / 3), fill::ones)
    * 2;

    x.submat(2 * floor(n / 3), 0, n - 1, 0) = mat(floor(n / 3), 1, fill::ones)
    * (-1);
    x.submat(2 * floor(n / 3), 1, n - 1, 1) = mat(floor(n / 3), 1, fill::ones)
    * (-1);

    y.subvec(2 * floor(n / 3), n - 1) = vec(n - 2 * floor(n / 3), fill::ones)
    * 3;

    // add noise
    x = x + mat(n, m, fill::randu) * step;

    mat xcopy(x);
    vec ycopy(y);

    vector<int> myvector;

    // set some values:
    for (int i = 0; i < n; ++i)
        myvector.push_back(i);

    // using built-in random generator:
    random_shuffle(myvector.begin(), myvector.end());

    // using myrandom:
    //random_shuffle ( myvector.begin(), myvector.end(), myrandom);

    // print out content:

    for (int i = 0; i < n; ++i) {
        x.submat(i, 0, i, m - 1) = xcopy.submat(myvector[i], 0, myvector[i],
                                                m - 1);

        y.subvec(i, i) = ycopy.subvec(myvector[i], myvector[i]);
    }

    tuple<mat, vec> result(x, y);
    return result;
}


double OLaRank_old::recomputeGradient(int i, int y){



    double grad=0;


    // update gradient g_i(y)
    grad-=this->loss((*this->S[i]->y).row(y),(*this->S[i]->y).row(this->S[i]->label));

    for (int j=0; j<this->S.size(); ++j) {
        for (int yhat=0; yhat<this->S[i]->x->n_rows; ++yhat) {
            grad-=(*this->S[j]->beta)(yhat)*this->kernel_fast(*this->S[i]->x,*this->S[i]->y, (*this->S[i]->y).row(y)(0),(*this->S[i]).frameNumber, *this->S[j]->x, *this->S[j]->y,(*this->S[j]->y).row(yhat)(0),(*this->S[j]).frameNumber);
        }
    }


    return grad;

}


void OLaRank_old::deleteKernelValues(int frameNumber){

    int f1=0;
    int f2=0;

    for (int i=0; i<this->S.size(); i++) {
        int frameNumber_2=this->S[i]->frameNumber;

        if (frameNumber<=frameNumber_2) {
            f1=frameNumber;
            f2=frameNumber_2;
        }else{
            f2=frameNumber;
            f1=frameNumber_2;
        }

        // get the key
        Key key(f1,f2);

        auto it=this->kern.find(key);

        if (it!=this->kern.end()) {
            // if the key is present. Delete the matrix associated with it from the heap

            //and then from the unordered map
            this->kern.erase(key);
        }

    }

}


std::ostream& operator<<(std::ostream &strm,const  OLaRank_old &s) {

    strm<<"OLaRank parameters: \n";
    strm<<"C                 : "<<s.parameters.C<<"\n";
    strm<<"n_R               : "<<s.parameters.n_R<<"\n";
    strm<<"n_O               : "<<s.parameters.n_O<<"\n";
    strm<<"B                 : "<<s.B<<"\n";
    strm<<"Kernel: \n"<<s.svm_kernel->getInfo()<<"\n";

    return strm;

}
OLaRank_old::~OLaRank_old(){

    {

        // Delete every memory associated with all the matrices
        for (int i=0; i<this->S.size(); i++) {
            deleteKernelValues(i);
        }

        for (int j = 0; j < this->S.size(); ++j) {
            supportData* s=S[j];
            delete s;
        }

        this->S.clear();

        this->clear();
        delete svm_kernel;
    }
}
