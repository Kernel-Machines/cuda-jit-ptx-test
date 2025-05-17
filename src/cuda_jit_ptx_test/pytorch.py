from cuda.core.experimental import Device, Stream, LaunchConfig, launch, Program, ProgramOptions

import torch
import sys

import threading
from typing import Dict, Any, Optional

class Singleton(type):
    """Metaclass for creating singleton classes."""
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
class DeviceModuleMap(metaclass=Singleton):
    """Singleton class mapping device IDs to lazily loaded modules."""
    
    def __init__(self):
        self._modules: Dict[int, Any] = {}  # device_id -> module
        self._lock = threading.Lock()

    def get_module(self, device_id: int) -> Any:
        if device_id < 0:
            raise ValueError(f"Invalid device ID: {device_id}")

        with self._lock:
            if device_id not in self._modules:
                self._modules[device_id] = self._load_module(device_id)
            return self._modules[device_id]

    def _load_module(self, device_id: int) -> Any:
        # Placeholder: Replace with actual module loading (e.g., CUDA PTX/cubin)
        return f"Module_for_device_{device_id}"  # Dummy module object

    def clear(self, device_id: Optional[int] = None) -> None:
        """
        Clear module(s) for a specific device or all devices.
        
        Args:
            device_id: Device ID to clear, or None to clear all.
        """
        with self._lock:
            if device_id is None:
                self._modules.clear()
            elif device_id in self._modules:
                del self._modules[device_id]

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

    device = Device(pt_device_id)

    device.set_current()
    stream = PyTorchStreamWrapper(pt_stream)

    # set cuda.core device based on this device


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