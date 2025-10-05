#ifndef DGEMM_BLAS_H
#define DGEMM_BLAS_H

#include "dgemm.h"

class DGEMM_blas : public DGEMM {
public:
    void compute(const Mat& A, const Mat& B, Mat& C) override;
    string name() override;
};

#endif // DGEMM_BLAS_H
