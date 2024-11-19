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
print("times:",tHpsi1,tHpsi2)

print("close:",np.allclose(Hpsi1,Hpsi2))

