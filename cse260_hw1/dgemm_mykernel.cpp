#include "dgemm_mykernel.h"
#include "parameters.h"
#include "pack.h"
#include "kernel.h"

#include <stdexcept>
#include <cstdlib> // for posix_memalign

void DGEMM_mykernel::compute(const Mat& A, const Mat& B, Mat& C) {
    int m = A.rows();
    int k = A.cols();
    int n = B.cols();

    my_dgemm(m, n, k, A.data(), k, B.data(), n, C.data(), n);
}

string DGEMM_mykernel::name() {
    return "my_kernel";
}

void DGEMM_mykernel::my_dgemm(
        int    m,
        int    n,
        int    k,
        const double* __restrict__ XA,
        int    lda,
        const double* __restrict__ XB,
        int    ldb,
        double *C,       
        int    ldc       
        )
{
    int    ic, ib, jc, jb, pc, pb;
    double *packA = nullptr, *packB = nullptr;

    posix_memalign((void**)&packA, 64, sizeof(double) * (( (param_mc + param_mr - 1)/param_mr ) * param_mr) * param_kc);
    posix_memalign((void**)&packB, 64, sizeof(double) * param_kc * ( ( (param_nc + param_nr - 1)/param_nr ) * param_nr ));

    for ( jc = 0; jc < m; jc += param_nc ) {              // 5-th loop around micro-kernel
        jb = min( m - jc, param_nc );
        for ( pc = 0; pc < k; pc += param_kc ) {          // 4-th loop around micro-kernel
            pb = min( k - pc, param_kc );
            
            pack_B_panel_KcNr(packB, XB, ldb, pb, jb, pc, jc, param_nr);

            for ( ic = 0; ic < n; ic += param_mc ) {        // 3-rd loop around micro-kernel
                ib = min( n - ic, param_mc );

                pack_A_panel_MrKc(packA, XA, lda, ib, pb, ic, pc, param_mr);

                // Implement your macro-kernel here
                my_macro_kernel(
                        ib,
                        jb,
                        pb,
                        packA,
                        packB,
                        &C[ ic * ldc + jc ], 
                        ldc
                        );
            }                                               // End 3.rd loop around micro-kernel
        }                                                 // End 4.th loop around micro-kernel
    }                                                     // End 5.th loop around micro-kernel
    free(packA);
    free(packB);
}

// Implement your macro-kernel here
void DGEMM_mykernel::my_macro_kernel(
        int    ib,
        int    jb,
        int    pb,
        const double* __restrict__ packA,
        const double* __restrict__ packB,
        double * C,
        int    ldc
        )
{
    int    i, j;

    for ( i = 0; i < ib; i += param_mr ) {                      // 2-th loop around micro-kernel
        int mr_eff = min(param_mr, ib - i);
        const double* a_sub = packA + ((i / param_mr) * (param_mr * pb));

        for ( j = 0; j < jb; j += param_nr ) {                  // 1-th loop around micro-kernel
            int nr_eff = min(param_nr, jb - j);
            const double* b_sub = packB + ((j / param_nr) * (param_nr * pb));
            double* c_sub = &C[i * ldc + j];
                my_dgemm_simulate_isters (
                            pb,
                            mr_eff,
                            nr_eff,
                            a_sub,          // assumes sq matrix, otherwise use lda
                            b_sub,
                            c_sub,                
                            ldc
                            );
        }                                                       // 1-th loop around micro-kernel
    }
}