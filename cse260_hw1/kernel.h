#pragma once
#include <cstddef>

void my_dgemm_ukr(int kc,
                  int mr,
                  int nr,
                  const double *__restrict__ a,
                  const double *__restrict__ b,
                  double *c,
                  int ldc);

void my_dgemm_simulate_registers(int kc,
                                 int mr, // should be 4
                                 int nr, // should be 4
                                 const double* __restrict__ a,
                                 const double* __restrict__ b,
                                 double *c,
                                 int ldc);

void my_dgemm_sve_8x4(int kc,
                     int mc, // shoule be 8
                     int nc, // shoule be 4
                      const double* __restrict__ A, const double* __restrict__ B,
                      double *C, int ldc);
