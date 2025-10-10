#ifndef DGEMM_MYKERNEL_H
#define DGEMM_MYKERNEL_H

#include "../dgemm/dgemm.h"

class DGEMM_mykernel : public DGEMM {
public:
    void compute(const Mat& A, const Mat& B, Mat& C) override;
    string name() override;
private:
    void my_dgemm(
        int    m,
        int    n,
        int    k,
        const double* __restrict__ XA,
        int    lda,
        const double* __restrict__ XB,
        int    ldb,
        double *C,       
        int    ldc       
        );

    void my_macro_kernel(
        int    ib,
        int    jb,
        int    pb,
        const double* __restrict__ packA,
        const double* __restrict__ packB,
        double * C,
        int    ldc
        );

};

#endif // DGEMM_MYKERNEL_H
