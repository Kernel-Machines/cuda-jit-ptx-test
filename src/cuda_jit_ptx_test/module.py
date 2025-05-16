
import sys

from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch, ObjectCode

import importlib.resources
from pathlib import Path

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