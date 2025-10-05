#ifndef DGEMM_BLISLAB_H
#define DGEMM_BLISLAB_H

#include "dgemm.h"

class DGEMM_blislab : public DGEMM {
public:
    void compute(const Mat& A, const Mat& B, Mat& C) override;
    string name() override;

private:
    void bl_dgemm(int m, int n, int k, const double *A, int lda, const double *B, int ldb, double *C, int ldc);
    void bl_macro_kernel(int ib, int jb, int pb, const double *packA, const double *packB, double *C, int ldc);
    void bl_dgemm_ukr(int kc, int mr, int nr, const double *a, const double *b, double *c, int ldc);
};

#endif // DGEMM_BLISLAB_H
