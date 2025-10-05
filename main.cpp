#include "dgemm/dgemm.h"
#include "dgemm/dgemm_naive.h"
#include "dgemm/dgemm_blas.h"
#include "dgemm/dgemm_blislab.h"
#include "cse260_hw1/dgemm_mykernel.h"
#include "matrix/matrix.h"
#include "utils/cmdLine.h"
#include <float.h>
#include <chrono>
#include <iostream>

#include "utils/dgemmChecker.h"

using namespace std;

namespace Setup {

    bool help = false;
    bool verb = false;
    bool noverif = false;
    bool debug = false;
    int size, seed, reps;
    std::string kernel;

    void usage() {
        std::cerr << "Usage:" << std::endl;
        std::cerr << "\t-h, --help" << std::endl;
        std::cerr << "\t-n=size, --size=size, [default 256]" << std::endl;
        std::cerr << "\t-r=reps, --reps=reps, [default 100]" << std::endl;
        std::cerr << "\t-v, --verb, [default not verbose]" << std::endl;
        std::cerr << "\t--noverif, [default verfication]" << std::endl;
        std::cerr << "\t-s=seed, --seed=seed, [default 1]" << std::endl;
        std::cerr << "\t-k=kernel, --kernel=kernel, [default naive_ijk]" << std::endl;
        std::cerr << "\t-d, --debug" << std::endl;
    }

    void summary() {
        std::cout << "runtime configuration summary:" << std::endl;
        std::cout << "\tsize: " << size << std::endl;
        std::cout << "\treps: " << reps << std::endl;
        std::cout << "\tverb: " << verb << std::endl;
        std::cout << "\tnoverif: " << noverif << std::endl;
        std::cout << "\tseed: " << seed << std::endl;
        std::cout << "\tkernel: " << kernel << std::endl;
        std::cout << "\tdebug: " << debug << std::endl;
    }

    void doSetup(CommandLineOptions &args) {
        help = args.help();
        if (help)  { usage(); std::exit(0); }
        if (args.illegal_present())  { usage(); std::exit(2); }
        verb = args.verbose();
        noverif = args.noverif();  debug = args.get_debug();
        size = args.size_n();  seed = args.get_seed();
        reps = args.reps();  kernel = args.kernel();
        summary();
    }
}

// ----------------------
// main
// ----------------------
int main(int argc, char *argv[]){
  Mat A;
  Mat B;
  Mat C;

  // ----------------------
  // set things up
  // ----------------------
  CommandLineOptions args(argc, argv);
  Setup::doSetup(args);
  const unsigned n = Setup::size;
  const unsigned m = Setup::size;
  const unsigned k = Setup::size;
  Mat::setSeed(Setup::seed);

  // allocate memory for matrices
  A.reserve(m * k); A.resize(m, k, 0.0);
  B.reserve(k * n); B.resize(k, n, 0.0);
  C.reserve(m * n);  C.resize(m, n, 0.0);

  if (Setup::debug){
    //    A.setSeq();
    //    B.setIdent();
    A.setRand();
    B.setRand();
    //    B.setUR(1.0/10);
    //    B.setRand();
    //A.setUR(1000000);
    //    B.setLL(.10);
    //A.setSeq();
    //    B.setRand();
  }else{
    A.setRand();
    B.setRand();
  }

    string kernel = Setup::kernel;
    std::shared_ptr<DGEMM> dgemm;
    if (kernel == "naive_ijk") {
        dgemm = std::make_shared<DGEMM_naive>();
    } else if (kernel == "openblas") {
        dgemm = std::make_shared<DGEMM_blas>();
    } else if (kernel == "blislab") {
        dgemm = std::make_shared<DGEMM_blislab>();
    } else if (kernel == "my_kernel") {
        dgemm = std::make_shared<DGEMM_mykernel>();
    } else {
        cerr << "Unknown kernel \"" << kernel << "\" for DGEMM interface\n";
        exit(1);
    }

    // ====================
    // check the correctness and performance of your implementation
    // ====================

    dgemmChecker checker(dgemm, Setup::size);

    if (Setup::noverif == 0) {
        cout << endl << "start correctness test..." << endl;
        bool success = checker.correctness(A, B, C, Setup::debug);
        if (!success)  exit(0);
        cout << "pass correctness test..." << endl;
    }
    if (Setup::debug)  exit(0);

    cout << endl << "start performance test..." << endl << endl;

    // warm up the cache
    checker.performance(A, B, C, 10);

    auto t_start = chrono::high_resolution_clock::now();
    checker.performance(A, B, C, Setup::reps);
    auto t_end  = chrono::high_resolution_clock::now();

    // print out the performance statistics

    double Gflops =
        (2.0 * (unsigned long)n * (unsigned long)m * (unsigned long)k) / 1.0e9;

    auto elapsed_time_sec =
        (std::chrono::duration<double, std::milli>(t_end - t_start).count()) / 1000.0;

    if (Setup::verb) {
        cout << "elapsed time = " << elapsed_time_sec << "\n";
        printf("Type (n=%d, reps = %d) %s\n", n, Setup::reps, dgemm->name().c_str());
        printf("GFlops per loop = %6.5f\n", Gflops);
        printf("Time(s)         = %5.4f\n", elapsed_time_sec);
        printf("GF/sec          = %5.4f\n",
            Setup::reps * Gflops / ((double)elapsed_time_sec));
    } else {
        printf("%-15s %4d %5.4f\n",
            dgemm->name().c_str(), n,
            Setup::reps * Gflops / ((double)elapsed_time_sec));
    }

    // unique_ptr automatically cleans up memory
    exit(0);
}
