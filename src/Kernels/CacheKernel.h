

#ifndef CACHED_KERNEL_H_
#define CACHED_KERNEL_H_

#include "Kernel.h"
#include "armadillo"
#include "unordered_map"
namespace cache {
struct Key {
    int row1;
    int row2;

    Key(int row1_, int row2_) {
        this->row1 = row1_;
        this->row2 = row2_;
    }
};

struct KeyHash {
    std::size_t operator()(const Key &k) const {
        return std::hash<int>()(k.row1 * k.row2);
    }
};

struct KeyEqual {
    bool operator()(const Key &lhs, const Key &rhs) const {
        return lhs.row1 == rhs.row1 && lhs.row2 == rhs.row2;
    }
};

}


class CachedKernel {

    Kernel *kernel;

    std::unordered_map<cache::Key, double, cache::KeyHash, cache::KeyEqual> kern;

  public:
CachedKernel(Kernel *kernel_) : kernel(kernel_){};

    double calculate( arma::mat &x1, int r1,  arma::mat &x2,
                           int r2);

    ~CachedKernel() { delete kernel; }
};

#endif
