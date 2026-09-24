from triton.tools.tensor_descriptor import TensorDescriptor
from .h100_tle_kernel import _tle_h100_kernel
from .tle_decode import _tle_decode_kernel
from .host import make_entry
def entry_for(config=None,fast=True):
    return make_entry('tle',_tle_h100_kernel,_tle_decode_kernel,TensorDescriptor,None,config,fast)
forgetting_attention=entry_for()
