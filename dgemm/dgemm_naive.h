#ifndef DGEMM_NAIVE_H
#define DGEMM_NAIVE_H

#include "dgemm.h"

class DGEMM_naive : public DGEMM {
public:
    void compute(const Mat& A, const Mat& B, Mat& C) override;
    string name() override;
};

#endif // DGEMM_NAIVE_H
