#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]
#include "parameters.h"

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
