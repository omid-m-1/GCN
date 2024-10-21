import torch as th
import torch.utils.dlpack
from . import graphpy as gpk # Import CUDA kernel
# Kernel API
def spmmv(input, res1, rowPtr, colInd, values, degrees, V, F_in):
    # Convert tensors to DLPack format
    input_dl = th.utils.dlpack.to_dlpack(input)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    rowPtr_dl = th.utils.dlpack.to_dlpack(rowPtr)
    colInd_dl = th.utils.dlpack.to_dlpack(colInd)
    values_dl = th.utils.dlpack.to_dlpack(values)
    degrees_dl = th.utils.dlpack.to_dlpack(degrees)
    # Run SpMMv using Cuda Kernel
    gpk.spmmv(input_dl, res_dl1, rowPtr_dl, colInd_dl, values_dl, degrees_dl, V, F_in)
    return res1
