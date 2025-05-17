from cuda.core.experimental import Device, Stream, LaunchConfig, launch, Program, ProgramOptions

import torch
import sys

class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)  # Return format required by CUDA Python

def test():
    code = r"""
        extern "C"
        __global__ 
        void 
        hello(
            float *A
        ) {
            float v = *A;

            printf("hello from device: %f\n", v);
            *A = v+1;
        }
        """
    
    # use torch stream to get device for the user
    pt_stream = torch.cuda.current_stream()
    pt_device = pt_stream.device
    pt_device_id = pt_device.index

    # set cuda.core device based on this device
    device = Device(pt_device_id)

    device.set_current()
    stream = PyTorchStreamWrapper(pt_stream)

    arch = "".join(f"{i}" for i in device.compute_capability)
    print(f'arch:{arch}')

    module_ptx = \
        Program(
            code, 
            code_type="c++", 
            options= \
                ProgramOptions(
                    std="c++17", 
                    arch=f"sm_{arch}"
                )
        ).compile("cubin", logs=sys.stdout)
    
    ker = module_ptx.get_kernel('hello')

    size = 64
    dtype = torch.float32

    a = torch.rand(size, dtype=dtype, device=pt_device)
    print(a)

    block = 1
    # grid = int((size + block - 1) // block)
    grid = int(1)
    config = LaunchConfig(grid=grid, block=block)
    ker_args = (a.data_ptr(),)

    # launch kernel on PyTorch's stream
    launch(stream, config, ker, *ker_args)
    print(a)