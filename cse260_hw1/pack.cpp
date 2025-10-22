#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifndef ALIGN64
#define ALIGN64 64
#endif

void* aligned_malloc(size_t bytes) {
    void* p = NULL;
    if (posix_memalign(&p, ALIGN64, bytes) != 0) abort();
    return p;
}


/*  Pack A_panel: size ib x pb taken from A(ic: ic+ib, pc: pc+pb), row major
    Output layout: size Mr x pb
    If tail <Mr or <pb, zero-pad
*/
void pack_A_panel_MrKc(double *__restrict__ A_pack,
                       const double *__restrict__ A, int lda,
                       int ib, int pb,
                       int ic, int pc,
                       int Mr)
{
    int ir = 0;
    for (; ir + Mr <= ib; ir += Mr)
    {
        // micro-panel size -> Mr x pb
        for (int k = 0; k < pb; k++)
        {
            const double *a_col = A + (ic + ir) * lda + (pc + k);
            for (int i = 0; i < Mr; i++)
            {
                A_pack[i] = a_col[i * lda];
            }
            A_pack += Mr;
        }
    }

    // tail rows (<Mr)
    if (ir < ib)
    {
        int tail = ib - ir;
        for (int k = 0; k < pb; k++)
        {
            const double *a_col = A + (ic + ir) * lda + (pc + k);
            int i = 0;
            for (; i < tail; i++)
                A_pack[i] = a_col[i * lda];
            for (; i < Mr; i++)
                A_pack[i] = 0.0;
            A_pack += Mr;
        }
    }
}

/*  Pack B_panel: size pb x jb taken from B(pc: pc+pb, jc: jc + jb), row major
    Output layout: size pb x Nr
    If tail < Nr or <pb, zero pad
*/
void pack_B_panel_KcNr(double *__restrict__ B_pack,
                       const double *__restrict__ B, int ldb,
                       int pb, int jb,  // actual size of B_panel
                       int pc, int jc,  // top-left in B
                       int Nr)  // micro-tile cols
{
    int jr = 0;
    for(; jr + Nr <= jb; jr += Nr){
        for(int k = 0; k < pb; k++){
            const double* b_row = B + (pc + k) * ldb + (jc + jr);
            memcpy(B_pack, b_row, sizeof(double) * Nr);
            B_pack += Nr;
        }
    }

    // tail cols ( < Nr )
    if (jr < jb) {
        int tail = jb - jr;
        for (int k = 0; k < pb; ++k) {
            const double* b_row = B + (pc + k) * ldb + (jc + jr);
            int j = 0;
            for (; j < tail; ++j) B_pack[j] = b_row[j];
            for (; j < Nr;   ++j) B_pack[j] = 0.0;  // pad
            B_pack += Nr;
        }
    }
}