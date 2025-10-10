#pragma once
#include <cstddef>


void my_dgemm_ukr( int    kc,
                                  int    mr, 
                                  int    nr, 
                                  const double* __restrict__ a,
                                  const double* __restrict__ b,
                                  double *c,
                                  int ldc);
                                  
void my_dgemm_simulate_registers( int    kc,
                                  int    mr, // should be 4
                                  int    nr, // should be 4
                                  const double* __restrict__ a,
                                  const double* __restrict__ b,
                                  double *c,
                                  int ldc);