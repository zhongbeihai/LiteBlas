#include "dgemmChecker.h"
#include <cblas.h>
#include <limits>
#include <iostream>

using namespace std;

bool dgemmChecker::correctness(Mat & A, Mat & B, Mat & C, bool debug) {

    this->dgemm->compute(A, B, C);

    if (debug && m <= 10) {
        cout << endl << "Matrix C:" << endl;
        C.print();
    }

    // calculate the error limit for DGEMM

    Mat Aabs = A;  Aabs.absVal();
    Mat Babs = B;  Babs.absVal();
    Mat Climit(m, n);

    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        m, n, k, 
        n * 3.0 * std::numeric_limits<double>::epsilon(),  // alpha
        Aabs.data(), A.cols(), Babs.data(), B.cols(), 
        0.0,  // beta
        Climit.data(), Climit.cols()
    );
    Climit.absVal();

    Mat Cref = C;
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        m, n, k, 
        -1.0,  // alpha
        A.data(), A.cols(), B.data(), B.cols(), 
        1.0,  // beta: scaling factor for matrix C
        Cref.data(), Cref.cols()
    );

    if (debug && m <= 10) {
        cout << endl << "diff Matrix C:" << endl;
        Cref.print();
    }
    Cref.absVal();

    int row, col;  double val;
    if (Cref.lessThan(Climit, val, row, col)) {
        fprintf(
            stderr, "out of error bounds (%d, %d) C=%4.5f diff=%4.5e\n", 
            row, col, C.data()[row * C.cols() + col], val
        );
        return false;
    }

    return true;
}

void dgemmChecker::performance(Mat & A, Mat & B, Mat & C, int reps) {
    for (int rep = 0; rep < reps; ++rep)
        this->dgemm->compute(A, B, C);
}
