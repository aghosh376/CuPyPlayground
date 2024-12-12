import os
import numpy as np
from util import npu
import operatornD
import operator1D
from ttns2 import contraction
import time
print("USE",os.environ["OMP_NUM_THREADS"],"threads")
MAX_MEM_IN_GB = 220 

nDim = 5
nBas = 3
nSum = 10
assert nBas**nDim * 8 * (3 +  int(os.environ["OMP_NUM_THREADS"])) * 1e-9 < MAX_MEM_IN_GB
Ns = [nBas] * nDim
Hop = operatornD.operatorSumOfProduct(nDim=nDim, nSum=nSum)
for iSum in range(Hop.nSum):
    for iDim in range(Hop.nDim):
        Hop[iDim, iSum] = operator1D.general(str=f"{iDim}_{iSum}")
        #Hop[iDim, iSum].mat = npu.randomSymmetric(Ns[iDim])
        Hop[iDim, iSum].mat = np.random.rand(Ns[iDim],Ns[iDim])
Hop.obtainMultiplyOp(nBases=Ns)

Hvec = contraction.getHvec(Hop, Ns, float)

psi = np.random.rand(*Ns)
psi = psi.ravel()

tStart = time.time()
Hpsi1 = Hop @ psi # python code for tensor contraction on CPU
tHpsi1 = time.time() - tStart

Hpsi2 = np.empty_like(psi)

tStart = time.time()
Hvec.dot(psi, Hpsi2, add=False) # c++ code for tensor contraction on CPU
tHpsi2 = time.time() - tStart
#print("times:",tHpsi1,tHpsi2)

#print("close:",np.allclose(Hpsi1,Hpsi2))

def benchmark_tensor_contraction_with_plan(n_runs=10):
    python_times = []
    cpp_times = []
    plan_times = []
    is_close = True

    for _ in range(n_runs):
        # Python tensor contraction
        t_start = time.time()
        Hpsi1 = Hop @ psi
        t_python = time.time() - t_start
        python_times.append(t_python)

        # C++ tensor contraction
        Hpsi2 = np.empty_like(psi)
        t_start = time.time()
        Hvec.dot(psi, Hpsi2, add=False)
        t_cpp = time.time() - t_start
        cpp_times.append(t_cpp)

        # Contraction with plan
        Hpsi3 = np.empty_like(psi)
        t_start = time.time()
        contraction.contract_with_plan(plan, psi, Hpsi3)
        t_plan = time.time() - t_start
        plan_times.append(t_plan)

        # Check if results match
        is_close = is_close and np.allclose(Hpsi1, Hpsi2) and np.allclose(Hpsi1, Hpsi3)

    # Print benchmark results
    print("\nBenchmark Results:")
    print(f"Python contraction (avg): {np.mean(python_times):.5f}s")
    print(f"C++ contraction (avg):   {np.mean(cpp_times):.5f}s")
    print(f"Plan contraction (avg):  {np.mean(plan_times):.5f}s")
    print(f"C++ Speedup over Python: {np.mean(python_times) / np.mean(cpp_times):.2f}x")
    print(f"Plan Speedup over Python: {np.mean(python_times) / np.mean(plan_times):.2f}x")
    print(f"Results match: {is_close}")

    return python_times, cpp_times, plan_times, is_close

# Run the benchmark
benchmark_tensor_contraction_with_plan(n_runs=10)

