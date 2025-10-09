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
        const double *XA,
        int    lda,
        const double *XB,
        int    ldb,
        double *C,       
        int    ldc       
        )
{
    int    ic, ib, jc, jb, pc, pb;
    const double *packA, *packB;

    // Using NOPACK option for simplicity
    // #define NOPACK
    double* A_pack_buf = nullptr;
    posix_memalign((void**)&A_pack_buf, 64, sizeof(double) * (( (param_mc + param_mr - 1)/param_mr ) * param_mr) * param_kc);
    double* B_pack_buf = nullptr;
    posix_memalign((void**)&B_pack_buf, 64, sizeof(double) * param_kc * ( ( (param_nc + param_nr - 1)/param_nr ) * param_nr ));

    for ( ic = 0; ic < m; ic += param_mc ) {              // 5-th loop around micro-kernel
        ib = min( m - ic, param_mc );
        for ( pc = 0; pc < k; pc += param_kc ) {          // 4-th loop around micro-kernel
            pb = min( k - pc, param_kc );
            
            #ifdef NOPACK
            packA = &XA[pc + ic * lda ];
            #else
            pack_A_panel_MrKc(A_pack_buf, XA, lda, ib, pb, ic, pc, param_mr);
            #endif

            for ( jc = 0; jc < n; jc += param_nc ) {        // 3-rd loop around micro-kernel
                jb = min( n - jc, param_nc );

                #ifdef NOPACK
                packB = &XB[ldb * pc + jc ];
                #else
                pack_B_panel_KcNr(B_pack_buf, XB, ldb, pb, jb, pc, jc, param_nr);
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
                                  const double *a,
                                  const double *b,
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
    for ( l = 0; l < kc; ++l ) {                 
        for ( i = 0; i < mr; ++i ) { 
            double as = a(i, l, ldc);
            for ( j = 0; j < nr; ++j ) { 
                cloc[i][j] +=  as * b(l, j, ldc);
            }
        }
    }
    
    // Store local array back to C
    for (i = 0; i < mr; ++i) {
        for (j = 0; j < nr; ++j) {
            c(i, j, ldc) = cloc[i][j];
        }
    }
}

// Implement your macro-kernel here
void DGEMM_mykernel::my_macro_kernel(
        int    ib,
        int    jb,
        int    pb,
        const double * packA,
        const double * packB,
        double * C,
        int    ldc
        )
{
    int    i, j;

    for ( i = 0; i < ib; i += param_mr ) {                      // 2-th loop around micro-kernel
        for ( j = 0; j < jb; j += param_nr ) {                  // 1-th loop around micro-kernel
            my_dgemm_ukr (
                        pb,
                        min(ib-i, param_mr),
                        min(jb-j, param_nr),
                        &packA[i * ldc],          // assumes sq matrix, otherwise use lda
                        &packB[j],                
                        &C[ i * ldc + j ],
                        ldc
                        );
        }                                                       // 1-th loop around micro-kernel
    }
}