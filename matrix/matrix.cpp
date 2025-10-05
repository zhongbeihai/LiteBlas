#include "matrix.h"

// Static member definitions
std::mt19937 Mat::gen_;
bool Mat::gen_initialized_ = false;

// Constructors
Mat::Mat() : rows_(0), cols_(0) {}

Mat::Mat(int rows, int cols) : rows_(rows), cols_(cols) {
    data_.resize(rows * cols, 0.0);
}

Mat::Mat(int rows, int cols, double init_val) : rows_(rows), cols_(cols) {
    data_.resize(rows * cols, init_val);
}

// Copy constructor and assignment
Mat::Mat(const Mat& other) : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {}

Mat& Mat::operator=(const Mat& other) {
    if (this != &other) {
        data_ = other.data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
    }
    return *this;
}

// Getters
int Mat::rows() const { return rows_; }
int Mat::cols() const { return cols_; }
int Mat::size() const { return rows_ * cols_; }
double* Mat::data() { return data_.data(); }
const double* Mat::data() const { return data_.data(); }

// // Element access
// double& Mat::operator()(int row, int col) {
//     return data_[row * cols_ + col];
// }

// const double& Mat::operator()(int row, int col) const {
//     return data_[row * cols_ + col];
// }

// Resize
void Mat::resize(int rows, int cols) {
    rows_ = rows;
    cols_ = cols;
    data_.resize(rows * cols);
}

void Mat::resize(int rows, int cols, double init_val) {
    rows_ = rows;
    cols_ = cols;
    data_.resize(rows * cols, init_val);
}

void Mat::reserve(int capacity) {
    data_.reserve(capacity);
}

// Static seed initialization
void Mat::setSeed(int seed) {
    gen_.seed(seed);
    gen_initialized_ = true;
}

// Matrix initialization methods (from utils.h)
void Mat::setIdent() {
    // if (rows_ != cols_) {
    //     throw std::invalid_argument("Identity matrix must be square");
    // }
    std::fill(data_.begin(), data_.end(), 0.0);
    for (int i = 0; i < rows_; i++) {
        data_[i * cols_ + i] = 1.0;
    }
}

void Mat::setUR(double val) {
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            data_[i * cols_ + j] = (i <= j) ? val : 0.0;
        }
    }
}

void Mat::setLL(double val) {
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            data_[i * cols_ + j] = (i >= j) ? val : 0.0;
        }
    }
}

void Mat::setRand() {
    if (!gen_initialized_) {
        setSeed();
    }
    std::uniform_real_distribution<double> dis(-1000.0, 1000.0);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            data_[i * cols_ + j] = dis(gen_);
        }
    }
}

void Mat::setSeq() {
    std::iota(data_.begin(), data_.end(), 1.0);
}

void Mat::setVal(double v) {
    std::fill(data_.begin(), data_.end(), v);
}

void Mat::absVal() {
    for (double& val : data_) {
        val = std::fabs(val);
    }
}

// Utility methods
int Mat::lessThan(const Mat& limit, double& val, int& row, int& col) const {
    // if (rows_ != limit.rows_ || cols_ != limit.cols_) {
    //     throw std::invalid_argument("Matrix dimensions must match for comparison");
    // }
    
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            int idx = i * cols_ + j;
            if (data_[idx] > limit.data_[idx]) {
                val = data_[idx];
                row = i;
                col = j;
                return 1;
            }
        }
    }
    return 0;
}

double Mat::maxMatDiff(const Mat& other) const {
    // if (rows_ != other.rows_ || cols_ != other.cols_) {
    //     throw std::invalid_argument("Matrix dimensions must match for comparison");
    // }
    
    double maxDiff = 0.0;
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            int idx = i * cols_ + j;
            double delta = std::fabs(data_[idx] - other.data_[idx]);
            if (delta > maxDiff) {
                maxDiff = delta;
            }
        }
    }
    return maxDiff;
}

void Mat::printDiff(const Mat& other, double diffThresh) const {
    // if (rows_ != other.rows_ || cols_ != other.cols_) {
    //     throw std::invalid_argument("Matrix dimensions must match for comparison");
    // }
    
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            int idx = i * cols_ + j;
            double delta = std::fabs(data_[idx] - other.data_[idx]);
            if (delta >= diffThresh) {
                printf("(%d, %d)\t %e\t %e \t%e\n", i, j, delta, data_[idx], other.data_[idx]);
            }
        }
    }
}

// Print methods
void Mat::print() const {
    print(rows_, cols_);
}

void Mat::print(int M, int N) const {
    const double* p = data_.data();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%4.3f\t", *p++);
        }
        printf("\n");
    }
}

// Static print methods for packed matrices (keeping original interface)
void Mat::printPackAMat(int M, int N, int mr, const double* X) {
    int Ml = M + M % mr;
    int Nl = N;
    std::vector<std::vector<double>> buf(Ml, std::vector<double>(Nl, -9999.99));
    
    const double* ptr = X;
    for (int i = 0; i < M; i += mr) {
        for (int j = 0; j < N; j++) {
            for (int ii = 0; ii < mr; ii++) {
                buf[i + ii][j] = *ptr++;
            }
        }
    }
    
    for (int i = 0; i < M; i += mr) {
        printf("----------\n");
        for (int ii = 0; ii < mr; ii++) {
            for (int j = 0; j < N; j++) {
                printf("%4.3f\t", buf[i + ii][j]);
            }
            printf("\n");
        }
    }
    printf("----------\n");
}

void Mat::printPackBMat(int M, int N, int nr, const double* X) {
    int Nl = N + N % nr;
    int Ml = M;
    std::vector<std::vector<double>> buf(Ml, std::vector<double>(Nl, -9999.99));

    const double* ptr = X;
    for (int j = 0; j < Nl; j += nr) {
        for (int i = 0; i < M; i++) {
            for (int jj = 0; jj < nr; jj++) {
                buf[i][j + jj] = *ptr++;
            }
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += nr) {
            printf("|");
            for (int jj = 0; jj < nr; jj++) {
                printf("%4.3f\t", buf[i][j + jj]);
            }
        }
        printf("\n");
    }
}

