#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]
#include "parameters.h"
#include <arm_sve.h>
#include <assert.h>
#include <stdint.h>

void my_dgemm_ukr( int    kc,
                                  int    mr, 
                                  int    nr, 
                                  const double* __restrict__ a,
                                  const double* __restrict__ b,
                                  double *c,
                                  int ldc)
{
    int l, j, i;
    double cloc[param_mr][param_nr] = {{0}};
    
    // Load C into local array
    for (i = 0; i < mr; ++i) {
        for (j = 0; j < nr; ++j) {
            cloc[i][j] = c(i, j, ldc);
        }
    }
    
    // Perform matrix multiplication
    const double* ap = a;
    const double* bp = b;
    for ( l = 0; l < kc; ++l ) {                 
        for ( i = 0; i < mr; ++i ) { 
            double as = ap[i];
            for ( j = 0; j < nr; ++j ) { 
                cloc[i][j] +=  as * bp[j];
            }
        }
        ap += param_mr;
        bp += param_nr;
    }
    
    // Store local array back to C
    for (i = 0; i < mr; ++i) {
        for (j = 0; j < nr; ++j) {
            c[i * ldc + j] = cloc[i][j];
        }
    }
}

// ** Simluate the regisetr**
// require mr = 4 and nr = 4
void my_dgemm_simulate_registers( int    kc,
                                  int    mr, // should be 4
                                  int    nr, // should be 4
                                  const double* __restrict__ a,
                                  const double* __restrict__ b,
                                  double *c,
                                  int ldc)
{
    double reg_c[4][4] = {{0.0}};

    const double* ap = a;
    const double* bp = b;

    for (int l = 0; l < kc; ++l) {
        double a0 = ap[0];
        double a1 = ap[1];
        double a2 = ap[2];
        double a3 = ap[3];
        
        // Unroll a little
        reg_c[0][0] += a0 * bp[0];
        reg_c[0][1] += a0 * bp[1];
        reg_c[0][2] += a0 * bp[2];
        reg_c[0][3] += a0 * bp[3];

        reg_c[1][0] += a1 * bp[0];
        reg_c[1][1] += a1 * bp[1];
        reg_c[1][2] += a1 * bp[2];
        reg_c[1][3] += a1 * bp[3];

        reg_c[2][0] += a2 * bp[0];
        reg_c[2][1] += a2 * bp[1];
        reg_c[2][2] += a2 * bp[2];
        reg_c[2][3] += a2 * bp[3];

        reg_c[3][0] += a3 * bp[0];
        reg_c[3][1] += a3 * bp[1];
        reg_c[3][2] += a3 * bp[2];
        reg_c[3][3] += a3 * bp[3];

        ap += param_mr;
        bp += param_nr;
    }

    for (int i = 0; i < mr; ++i) {
        for (int j = 0; j < nr; ++j) {
            c[i * ldc + j] += reg_c[i][j];
        }
    }
}


void my_dgemm_sve_8x4(int kc,
                     int mr_eff, // shoule be 8
                     int nr_eff, // shoule be 4
                      const double* __restrict__ A, const double* __restrict__ B,
                      double *C, int ldc)
{
    assert(svcntd() == 4);
    const svbool_t pg = svptrue_b64();

    svfloat64_t c0 = svld1(pg, C + 0*ldc);
    svfloat64_t c1 = svld1(pg, C + 1*ldc);
    svfloat64_t c2 = svld1(pg, C + 2*ldc);
    svfloat64_t c3 = svld1(pg, C + 3*ldc);
    svfloat64_t c4 = svld1(pg, C + 4*ldc);
    svfloat64_t c5 = svld1(pg, C + 5*ldc);
    svfloat64_t c6 = svld1(pg, C + 6*ldc);
    svfloat64_t c7 = svld1(pg, C + 7*ldc);

    const double* ap = A;
    const double* bp = B;
    for (int l = 0; l < kc; ++l) {
        svfloat64_t b = svld1(pg, bp); 

        svfloat64_t a0 = svdup_f64(ap[0]); c0 = svmla_f64_x(pg, c0, a0, b);
        svfloat64_t a1 = svdup_f64(ap[1]); c1 = svmla_f64_x(pg, c1, a1, b);
        svfloat64_t a2 = svdup_f64(ap[2]); c2 = svmla_f64_x(pg, c2, a2, b);
        svfloat64_t a3 = svdup_f64(ap[3]); c3 = svmla_f64_x(pg, c3, a3, b);
        svfloat64_t a4 = svdup_f64(ap[4]); c4 = svmla_f64_x(pg, c4, a4, b);
        svfloat64_t a5 = svdup_f64(ap[5]); c5 = svmla_f64_x(pg, c5, a5, b);
        svfloat64_t a6 = svdup_f64(ap[6]); c6 = svmla_f64_x(pg, c6, a6, b);
        svfloat64_t a7 = svdup_f64(ap[7]); c7 = svmla_f64_x(pg, c7, a7, b);

        ap += 8;  // param_mr
        bp += 4;  // param_nr
    }

    svst1(pg, C + 0*ldc, c0);
    svst1(pg, C + 1*ldc, c1);
    svst1(pg, C + 2*ldc, c2);
    svst1(pg, C + 3*ldc, c3);
    svst1(pg, C + 4*ldc, c4);
    svst1(pg, C + 5*ldc, c5);
    svst1(pg, C + 6*ldc, c6);
    svst1(pg, C + 7*ldc, c7);
}
