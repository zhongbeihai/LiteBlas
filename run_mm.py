import subprocess
import re

def run_mm_and_parse_output(kernel="my_kernel", N = 512):
    try:
        cmd = f'./mm --kernel="{kernel}" --size="{N}"'
        result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)

        output = result.stdout
        # print("=== Raw Output ===")
        # print(output)

        # Extract performance (last number after kernel name)
        perf_match = re.search(rf"{kernel}\s+\d+\s+([0-9.]+)", output)
        if perf_match:
            performance = float(perf_match.group(1))
        else:
            raise ValueError("Performance value not found in output.")

        print(f"Parsed size: {N}, performance: {performance} GFLOPS")
        return N, performance

    except subprocess.CalledProcessError as e:
        print("Command failed to execute.")
        print(e)
        return None, None
    except Exception as e:
        print("Error parsing output:", e)
        return None, None

if __name__ == "__main__":
    Ns = [32, 64, 128, 255, 256, 510, 512, 513, 768, 769, 1023, 1024, 1025, 1033, 2047, 2048, 2049]
    my_kernel_perf = {}
    # naive_kernel = {}
    # blas_kernel_perf = {}

    # run for my_kernel
    print("Running my_kernel benchmarks...")
    for N in Ns:
        size, performance = run_mm_and_parse_output(kernel="my_kernel", N=N)
        my_kernel_perf[N] = performance

    # run for naive_kernel
    # print("Running naive_kernel benchmarks...")
    # for N in Ns:
    #     size, performance = run_mm_and_parse_output(kernel="naive_ijk", N=N)
    #     naive_kernel[N] = performance

    # run for blas_kernel
    # print("Running openblas benchmarks...")
    # for N in Ns:
    #     size, performance = run_mm_and_parse_output(kernel="openblas", N=N)
    #     blas_kernel_perf[N] = performance