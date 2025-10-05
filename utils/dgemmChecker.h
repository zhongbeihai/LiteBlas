#ifndef DGEMM_CHECKER_H
#define DGEMM_CHECKER_H

#include "../matrix/matrix.h"
#include "../dgemm/dgemm.h"
#include <memory>

class dgemmChecker {

public:

    explicit dgemmChecker(std::shared_ptr<DGEMM> dgemm, unsigned sz) :
    dgemm(dgemm), m(sz), n(sz), k(sz) {}

    // Prevent copying to avoid issues with pointer ownership
    dgemmChecker(const dgemmChecker&) = delete;
    dgemmChecker& operator=(const dgemmChecker&) = delete;

    bool correctness(Mat & A, Mat & B, Mat & C, bool debug);
    void performance(Mat & A, Mat & B, Mat & C, int reps);

private:
    std::shared_ptr<DGEMM> dgemm;  // Non-owning pointer
    const unsigned m, n, k;

};

#endif // DGEMM_CHECKER_H
