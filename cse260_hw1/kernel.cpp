#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]
#include "parameters.h"
#include <arm_sve.h>

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
    const uint64_t vector_len = svcntd();
    const bool has_second_chunk = static_cast<uint64_t>(nr_eff) > vector_len;

    svbool_t pg_col0 = svwhilelt_b64(static_cast<uint64_t>(0), static_cast<uint64_t>(nr_eff));
    svbool_t pg_col1 = has_second_chunk
                           ? svwhilelt_b64(vector_len, static_cast<uint64_t>(nr_eff))
                           : svpfalse_b();

    svfloat64_t c0_0 = svdup_f64(0.0), c0_1 = svdup_f64(0.0);
    svfloat64_t c1_0 = svdup_f64(0.0), c1_1 = svdup_f64(0.0);
    svfloat64_t c2_0 = svdup_f64(0.0), c2_1 = svdup_f64(0.0);
    svfloat64_t c3_0 = svdup_f64(0.0), c3_1 = svdup_f64(0.0);

    if (mr_eff > 0) {
        c0_0 = svld1(pg_col0, C + 0 * ldc);
        if (has_second_chunk) {
            c0_1 = svld1(pg_col1, C + 0 * ldc + vector_len);
        }
    }
    if (mr_eff > 1) {
        c1_0 = svld1(pg_col0, C + 1 * ldc);
        if (has_second_chunk) {
            c1_1 = svld1(pg_col1, C + 1 * ldc + vector_len);
        }
    }
    if (mr_eff > 2) {
        c2_0 = svld1(pg_col0, C + 2 * ldc);
        if (has_second_chunk) {
            c2_1 = svld1(pg_col1, C + 2 * ldc + vector_len);
        }
    }
    if (mr_eff > 3) {
        c3_0 = svld1(pg_col0, C + 3 * ldc);
        if (has_second_chunk) {
            c3_1 = svld1(pg_col1, C + 3 * ldc + vector_len);
        }
    }

    const double* ap = A;
    const double* bp = B;

    for (int l = 0; l < kc; ++l) {
        svfloat64_t b0 = svld1(pg_col0, bp);
        svfloat64_t b1 = svdup_f64(0.0);
        if (has_second_chunk) {
            b1 = svld1(pg_col1, bp + vector_len);
        }

        if (mr_eff > 0) {
            svfloat64_t a0_bcast = svdup_f64(ap[0]);
            c0_0 = svmla_f64_m(pg_col0, c0_0, a0_bcast, b0);
            if (has_second_chunk) {
                c0_1 = svmla_f64_m(pg_col1, c0_1, a0_bcast, b1);
            }
        }
        if (mr_eff > 1) {
            svfloat64_t a1_bcast = svdup_f64(ap[1]);
            c1_0 = svmla_f64_m(pg_col0, c1_0, a1_bcast, b0);
            if (has_second_chunk) {
                c1_1 = svmla_f64_m(pg_col1, c1_1, a1_bcast, b1);
            }
        }
        if (mr_eff > 2) {
            svfloat64_t a2_bcast = svdup_f64(ap[2]);
            c2_0 = svmla_f64_m(pg_col0, c2_0, a2_bcast, b0);
            if (has_second_chunk) {
                c2_1 = svmla_f64_m(pg_col1, c2_1, a2_bcast, b1);
            }
        }
        if (mr_eff > 3) {
            svfloat64_t a3_bcast = svdup_f64(ap[3]);
            c3_0 = svmla_f64_m(pg_col0, c3_0, a3_bcast, b0);
            if (has_second_chunk) {
                c3_1 = svmla_f64_m(pg_col1, c3_1, a3_bcast, b1);
            }
        }

        ap += param_mr;
        bp += param_nr;
    }

    if (mr_eff > 0) {
        svst1(pg_col0, C + 0 * ldc, c0_0);
        if (has_second_chunk) {
            svst1(pg_col1, C + 0 * ldc + vector_len, c0_1);
        }
    }
    if (mr_eff > 1) {
        svst1(pg_col0, C + 1 * ldc, c1_0);
        if (has_second_chunk) {
            svst1(pg_col1, C + 1 * ldc + vector_len, c1_1);
        }
    }
    if (mr_eff > 2) {
        svst1(pg_col0, C + 2 * ldc, c2_0);
        if (has_second_chunk) {
            svst1(pg_col1, C + 2 * ldc + vector_len, c2_1);
        }
    }
    if (mr_eff > 3) {
        svst1(pg_col0, C + 3 * ldc, c3_0);
        if (has_second_chunk) {
            svst1(pg_col1, C + 3 * ldc + vector_len, c3_1);
        }
    }
}
