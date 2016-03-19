
#include "CacheKernel.h"
#include <cfloat>

double CachedKernel::calculate(arma::mat &x1, int row1, arma::mat &x2,
                               int row2) {

    // to save only half of the kernels
    if (row1 < row2) {
        return this->calculate(x1, row2, x2, row1);
    }

    // we can always assume that row1>=row2
    cache::Key key(row1, row2);

    auto it = (this->kern).find(key);

    double result = 0;

    if (it == (this->kern).end()) {
        // key is not found
        result = this->kernel->calculate(x1, row1, x2, row2);
        (this->kern).insert({key, result});
    } else {
        return it->second;
    }
    return result;
}

void CachedKernel::initialize(int n) {
    this->cached_kernel.ones(n, n);
}

double CachedKernel::calculateFast(arma::mat &x1, int row1, arma::mat &x2,
                                   int row2) {
    if (row1 > row2) {
        return this->calculate(x1, row2, x2, row1);
    }

    if (this->cached_kernel(row1, row2) != DBL_MAX) {
        return this->cached_kernel(row1, row2);
    } else {
        this->cached_kernel(row1, row2) =
            this->kernel->calculate(x1, row1, x2, row2);
        return this->cached_kernel(row1, row2);
    }
}
