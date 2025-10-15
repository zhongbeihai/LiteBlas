import os
import subprocess
import re
import itertools
import sys

PROJECT_DIR = '.'
PARAMETERS_HEADER = os.path.join(PROJECT_DIR, './cse260_hw1/parameters.h')
MAKE_COMMAND = 'make clean && make'
EXECUTABLE_PATH = os.path.join(PROJECT_DIR, 'mm')
MATRIX_SIZE = 512

MC_VALUES = [128, 256]
NC_VALUES = [128, 256]
KC_VALUES = [64, 96, 128, 192]

HEADER_TEMPLATE = """
#ifndef PARAMETERS_H
#define PARAMETERS_H

#define PARAM_MC {mc}
#define PARAM_NC {nc}
#define PARAM_KC {kc}

#define PARAM_MR 4
#define PARAM_NR 8

inline constexpr int param_mc = PARAM_MC;
inline constexpr int param_nc = PARAM_NC;
inline constexpr int param_kc = PARAM_KC;
inline constexpr int param_mr = PARAM_MR;
inline constexpr int param_nr = PARAM_NR;

#endif // PARAMETERS_H
"""

def update_parameters_header(mc, nc, kc):
    print(f"  -> Writing parameters.h: MC={mc}, NC={nc}, KC={kc}")
    try:
        content = HEADER_TEMPLATE.format(mc=mc, nc=nc, kc=kc)
        with open(PARAMETERS_HEADER, 'w') as f:
            f.write(content)
        return True
    except IOError as e:
        print(f"ERROR: Failed to write to file {PARAMETERS_HEADER}: {e}", file=sys.stderr)
        return False

def compile_project():
    print(f"  -> Compiling project (using '{MAKE_COMMAND}')...")
    try:
        result = subprocess.run(
            MAKE_COMMAND,
            shell=True,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print("\n--- COMPILATION FAILED! ---", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        print("STDERR:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        print("---------------------------\n", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"ERROR: Command '{MAKE_COMMAND}' not found. Please ensure 'make' is installed and in your PATH.", file=sys.stderr)
        return False

def run_benchmark():
    command = [EXECUTABLE_PATH, '-k', 'my_kernel', '-n', str(MATRIX_SIZE)]
    print(f"  -> Running benchmark: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Benchmark command failed with return code {e.returncode}", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"ERROR: Executable '{EXECUTABLE_PATH}' not found. Did the compilation succeed?", file=sys.stderr)
        return None

def parse_gflops(output):
    match = re.search(r"my_kernel_gflops:\s*(\d+\.?\d*)", output)
    if match:
        return float(match.group(1))
    
    print("  -> WARNING: Could not parse GFLOPS value from output.", file=sys.stderr)
    print("    Program output:", file=sys.stderr)
    print(output, file=sys.stderr)
    return None

def main():
    print("--- Starting DGEMM Parameter Auto-Tuning ---")

    if not os.path.exists(PARAMETERS_HEADER):
        print(f"\nERROR: Parameters file not found at '{PARAMETERS_HEADER}'", file=sys.stderr)
        print("Please update the 'PARAMETERS_HEADER' variable in this script.", file=sys.stderr)
        return

    param_combinations = list(itertools.product(MC_VALUES, NC_VALUES, KC_VALUES))
    total_runs = len(param_combinations)
    
    results = []
    
    for i, (mc, nc, kc) in enumerate(param_combinations):
        print(f"\n--- Run {i+1}/{total_runs}: Testing MC={mc}, NC={nc}, KC={kc} ---")

        if not update_parameters_header(mc, nc, kc):
            print("Stopping due to file write error.", file=sys.stderr)
            break
            
        if not compile_project():
            print("Stopping due to compilation failure.", file=sys.stderr)
            break
            
        output = run_benchmark()
        
        if output:
            gflops = parse_gflops(output)
            if gflops is not None:
                print(f"  -> Performance: {gflops:.4f} GFLOPS")
                results.append({'params': {'MC': mc, 'NC': nc, 'KC': kc}, 'gflops': gflops})

    if not results:
        print("\n--- Tuning process finished with no successful runs. ---")
        print("Please check for compilation or runtime errors.")
    else:
        best_result = max(results, key=lambda r: r['gflops'])
        
        print("\n\n--- Tuning Process Finished ---")
        print("===================================")
        print(f" Peak Performance: {best_result['gflops']:.4f} GFLOPS")
        print("Optimal Parameters:")
        print(f"  PARAM_MC = {best_result['params']['MC']}")
        print(f"  PARAM_NC = {best_result['params']['NC']}")
        print(f"  PARAM_KC = {best_result['params']['KC']}")
        print("===================================")

if __name__ == "__main__":
    main()