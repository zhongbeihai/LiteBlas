#ifndef DGEMM_H
#define DGEMM_H

#include "../matrix/matrix.h"

class DGEMM {
public:
    virtual ~DGEMM() = default;
    virtual void compute(const Mat& A, const Mat& B, Mat& C) = 0;
    virtual string name() = 0;
};

#endif // DGEMM_H
