#include "dgemm_blas.h"
#include <cblas.h>

void DGEMM_blas::compute(const Mat& A, const Mat& B, Mat& C) {
    int m = A.rows();
    int k = A.cols();
    int n = B.cols();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        m, n, k, 1.0, 
        A.data(), k, B.data(), n, 0.0, C.data(), n);
}

string DGEMM_blas::name() {
    return "openblas";
}
