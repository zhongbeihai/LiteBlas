#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <numeric>

using namespace std;

class Mat {
private:
    std::vector<double> data_;
    int rows_;
    int cols_;
    static std::mt19937 gen_;
    static bool gen_initialized_;

public:
    // Constructors
    Mat();
    Mat(int rows, int cols);
    Mat(int rows, int cols, double init_val);

    // Copy constructor and assignment
    Mat(const Mat& other);
    Mat& operator=(const Mat& other);

    // Getters
    int rows() const;
    int cols() const;
    int size() const;
    double* data();
    const double* data() const;

    // // Element access
    // double& operator()(int row, int col);
    // const double& operator()(int row, int col) const;

    // Resize
    void resize(int rows, int cols);
    void resize(int rows, int cols, double init_val);
    void reserve(int capacity);

    // Static seed initialization
    static void setSeed(int seed = 1);

    // Matrix initialization methods (from utils.h)
    void setIdent();
    void setUR(double val = 1.0);
    void setLL(double val = 1.0);
    void setRand();
    void setSeq();
    void setVal(double v);
    void absVal();

    // Utility methods
    int lessThan(const Mat& limit, double& val, int& row, int& col) const;
    double maxMatDiff(const Mat& other) const;
    void printDiff(const Mat& other, double diffThresh) const;

    // Print methods
    void print() const;
    void print(int M, int N) const;

    // Static print methods for packed matrices (keeping original interface)
    static void printPackAMat(int M, int N, int mr, const double* X);
    static void printPackBMat(int M, int N, int nr, const double* X);
};

#endif // MATRIX_H