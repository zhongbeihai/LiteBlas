// pack.h
#pragma once
#include <cstddef>

void pack_A_panel_MrKc(double *__restrict__ A_pack,
                       const double *__restrict__ A, int lda,
                       int ib, int pb,
                       int ic, int pc,
                       int Mr);

void pack_B_panel_KcNr(double *__restrict__ B_pack,
                       const double *__restrict__ B, int ldb,
                       int pb, int jb,  // actual size of B_panel
                       int pc, int jc,  // top-left in B
                       int Nr);  // micro-tile cols
