
import sys
import os
import logging

from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch, ObjectCode

import importlib.resources
from pathlib import Path

# from .paths import *

def read_local_ptx_data():
    """Read the uncompressed local.ptx file bundled with the package."""
    package = "cuda_jit_ptx_test.data"
    resource = "local.ptx"
    
    try:
        with importlib.resources.as_file(importlib.resources.files(package) / resource) as ptx_path:
            with open(ptx_path, "r", encoding="utf-8") as f:
                data = f.read()  # Read as text
                return data
    except FileNotFoundError:
        raise FileNotFoundError("local.ptx not found in package data directory")
    except UnicodeDecodeError:
        raise ValueError("local.ptx contains invalid UTF-8 content")
    
def read_local_ptx_data_bytes():
    """Read the uncompressed local.ptx file bundled with the package."""
    package = "cuda_jit_ptx_test.data"
    resource = "local.ptx"
    
    try:
        with importlib.resources.as_file(importlib.resources.files(package) / resource) as ptx_path:
            with open(ptx_path, "rb") as f:
                data = f.read()  # Read as text
                return data
    except FileNotFoundError:
        raise FileNotFoundError("local.ptx not found in package data directory")
    except UnicodeDecodeError:
        raise ValueError("local.ptx contains invalid UTF-8 content")
        
def get_top_level_repo_dir(dir) -> Path:
    # directory *beside* the package (wheel layout)
    wheel_copy = importlib.resources.files("cuda_jit_ptx_test").parent / dir
    # directory *beside* src/ (editable / repo checkout)
    repo_copy  = Path(__file__).resolve().parents[2] / dir

    for path in (repo_copy, wheel_copy):
        if path.is_dir():
            return path.resolve()

    raise FileNotFoundError("thirdparty directory not found")

def get_local_cuda_src_dir() -> Path:
    return get_top_level_repo_dir('csrc')

def get_include_local_cuda_dir() -> Path:
    return get_top_level_repo_dir('include')

def get_include_dir_cutlass() -> Path:
    return get_top_level_repo_dir('thirdparty') / 'cutlass/include'

def get_include_dir_cutlass_tools() -> Path:
    return get_top_level_repo_dir('thirdparty') / 'cutlass/tools/util/include'

def get_include_dir_cuda() -> Path:
    """Best-effort guess of the Toolkit’s <cuda>/include directory."""
    import os, shutil
    if os.getenv("CUDA_HOME"):
        return Path(os.environ["CUDA_HOME"]) / "include"
    # fall back to the directory that owns nvcc (works for most local installs)
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).parent.parent / "include"
    raise RuntimeError("Cannot find CUDA include directory")

def run_local():

    dev = Device()
    dev.set_current()
    s = dev.create_stream()

    arch = "".join(f"{i}" for i in dev.compute_capability)
    print(f'arch:{arch}')


    ptx_bytes = read_local_ptx_data_bytes()
    module_ptx_deserialized = ObjectCode.from_ptx(ptx_bytes)

    # Compile ptx to cubin
    module_cubin = \
        Program(
            module_ptx_deserialized.code.decode(), 
            code_type="ptx", 
            options= \
                ProgramOptions(
                    arch=f"sm_{arch}"
                )
        ).compile("cubin", logs=sys.stdout)

    # Store cubin to file
    with open('output.cubin', 'wb') as file:
        file.write(module_cubin.code)

    # read cubin from file
    modulde_cubin_deserialized = ObjectCode.from_cubin('output.cubin')

    kernel_function = modulde_cubin_deserialized.get_kernel("hello")
    print(f'num_registers: {kernel_function.attributes.num_regs()}')

    block = 1
    grid = int(1)
    config = LaunchConfig(grid=grid, block=block)
    kernel_args = ()
    launch(s, config, kernel_function, *kernel_args)
    s.sync()


def run_local_cuda():

    cuda_code_path = get_local_cuda_src_dir() / 'test.cu'
    with open(cuda_code_path, "r", encoding="utf-8") as f:
        code = f.read()  # Read as text

    dev = Device()
    dev.set_current()
    s = dev.create_stream()

    arch = "".join(f"{i}" for i in dev.compute_capability)
    print(f'arch:{arch}')

    module_ptx = \
        Program(
            code, 
            code_type="c++", 
            options= \
                ProgramOptions(
                    std="c++17", 
                    arch=f"sm_{arch}",
                    device_as_default_execution_space=True,
                    include_path=[
                        get_include_local_cuda_dir(),       # *.cuh
                        get_include_dir_cutlass(),          # main cutlass include
                        get_include_dir_cutlass_tools(),    # cutlass tools include
                        get_include_dir_cuda()              # cuda toolkit for <cuda/src/assert>, etc.. (dependency of cutlass)
                    ]
                )
        ).compile(
            "ptx", 
            logs=sys.stdout,
        )

    # print(module_ptx.code.decode())

    # Store ptx to file
    with open('output.ptx', 'wb') as file:
        file.write(module_ptx.code)

    # Read ptx from file
    with open('output.ptx', 'rb') as file:
        ptx_bytes = file.read()
        module_ptx_deserialized = ObjectCode.from_ptx(ptx_bytes)

    # Compile ptx to cubin
    module_cubin = \
        Program(
            module_ptx_deserialized.code.decode(), 
            code_type="ptx", 
            options= \
                ProgramOptions(
                    arch=f"sm_{arch}",
                    ptxas_options=['-O3']
                )
        ).compile("cubin", logs=sys.stdout)

    # Store cubin to file
    with open('output.cubin', 'wb') as file:
        file.write(module_cubin.code)

    # read cubin from file
    modulde_cubin_deserialized = ObjectCode.from_cubin('output.cubin')

    kernel_function = modulde_cubin_deserialized.get_kernel("kernel")
    print(f'num_registers: {kernel_function.attributes.num_regs()}')

    block = 1
    grid = int(1)
    config = LaunchConfig(grid=grid, block=block)
    kernel_args = ()
    launch(s, config, kernel_function, *kernel_args)
    s.sync()

def run():

    code = r"""
        extern "C"
        __global__ 
        void 
        hello() {
            printf("hello from device\n");
        }
        """

    dev = Device()
    dev.set_current()
    s = dev.create_stream()

    arch = "".join(f"{i}" for i in dev.compute_capability)
    print(f'arch:{arch}')

    # Compile cuda to ptx
    module_ptx = \
        Program(
            code, 
            code_type="c++", 
            options= \
                ProgramOptions(
                    std="c++17", 
                    arch=f"sm_{arch}"
                )
        ).compile("ptx", logs=sys.stdout)

    print(module_ptx.code.decode())

    # Store ptx to file
    with open('output.ptx', 'wb') as file:
        file.write(module_ptx.code)

    # Read ptx from file
    with open('output.ptx', 'rb') as file:
        ptx_bytes = file.read()
        module_ptx_deserialized = ObjectCode.from_ptx(ptx_bytes)

    # Compile ptx to cubin
    module_cubin = \
        Program(
            module_ptx_deserialized.code.decode(), 
            code_type="ptx", 
            options= \
                ProgramOptions(
                    arch=f"sm_{arch}"
                )
        ).compile("cubin", logs=sys.stdout)

    # Store cubin to file
    with open('output.cubin', 'wb') as file:
        file.write(module_cubin.code)

    # read cubin from file
    modulde_cubin_deserialized = ObjectCode.from_cubin('output.cubin')

    kernel_function = modulde_cubin_deserialized.get_kernel("hello")
    print(f'num_registers: {kernel_function.attributes.num_regs()}')

    block = 1
    grid = int(1)
    config = LaunchConfig(grid=grid, block=block)
    kernel_args = ()
    launch(s, config, kernel_function, *kernel_args)
    s.sync()