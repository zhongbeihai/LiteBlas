#include "dgemm_naive.h"

void DGEMM_naive::compute(const Mat& A, const Mat& B, Mat& C) {
    int m = A.rows();
    int k = A.cols();
    int n = B.cols();
    
    // Naive ijk matrix multiplication
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double c = 0.0;
            for (int x = 0; x < k; x++) {
                c = c + A.data()[i * k + x] * B.data()[x * n + j];
            }
            C.data()[i * n + j] = c;
        }
    }
}

string DGEMM_naive::name() {
    return "naive_ijk";
}
