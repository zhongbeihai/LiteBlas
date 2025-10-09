#include "dgemm_mykernel.h"
#include "parameters.h"
#include "pack.h"

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

    // Using NOPACK option for simplicity
    // #define NOPACK
    posix_memalign((void**)&packA, 64, sizeof(double) * (( (param_mc + param_mr - 1)/param_mr ) * param_mr) * param_kc);
    posix_memalign((void**)&packB, 64, sizeof(double) * param_kc * ( ( (param_nc + param_nr - 1)/param_nr ) * param_nr ));

    for ( ic = 0; ic < m; ic += param_mc ) {              // 5-th loop around micro-kernel
        ib = min( m - ic, param_mc );
        for ( pc = 0; pc < k; pc += param_kc ) {          // 4-th loop around micro-kernel
            pb = min( k - pc, param_kc );
            
            #ifdef NOPACK
            packA = &XA[pc + ic * lda ];
            #else
            pack_A_panel_MrKc(packA, XA, lda, ib, pb, ic, pc, param_mr);
            #endif

            for ( jc = 0; jc < n; jc += param_nc ) {        // 3-rd loop around micro-kernel
                jb = min( n - jc, param_nc );

                #ifdef NOPACK
                packB = &XB[ldb * pc + jc ];
                #else
                pack_B_panel_KcNr(packB, XB, ldb, pb, jb, pc, jc, param_nr);
                #endif

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

#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

//
// C-based microkernel (NOPACK version)
//
// Implement your micro-kernel here
void DGEMM_mykernel::my_dgemm_ukr( int    kc,
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
            my_dgemm_ukr (
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