
import sys
import os
import logging

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


def read_local_cuda():
    """Read the uncompressed local.ptx file bundled with the package."""
    package = "cuda_jit_ptx_test.cuda"
    resource = "test.cu"
    
    try:
        with importlib.resources.as_file(importlib.resources.files(package) / resource) as ptx_path:
            with open(ptx_path, "r", encoding="utf-8") as f:
                data = f.read()  # Read as text
                return data
    except FileNotFoundError:
        raise FileNotFoundError("local.ptx not found in package data directory")
    except UnicodeDecodeError:
        raise ValueError("local.ptx contains invalid UTF-8 content")

from importlib.resources import files

def get_resource_dir(resource_type):
    """Get stable string path to a resource directory (e.g., 'cuda', 'cuda.include', 'thirdparty')."""
    valid_types = {"cuda", "cuda.include", "data", "thirdparty"}
    if resource_type not in valid_types:
        raise ValueError(f"resource_type must be one of {valid_types}")
    
    if resource_type == "thirdparty":
        package = "thirdparty"  # Top-level thirdparty directory
    else:
        package = f"cuda_jit_ptx_test.{resource_type}"  # Subpackages under cuda_jit_ptx_test
    
    resource_path = files(package)
    if not resource_path.is_dir():
        raise FileNotFoundError(f"{resource_type} directory not found")
    
    # Handle MultiplexedPath by accessing the first path or converting to string
    if hasattr(resource_path, '_paths'):
        return str(resource_path._paths[0])
    return str(resource_path)

def get_cutlass_folder(package_name: str = "cuda_jit_ptx_test", folder_path: str = "thirdparty") -> Path | None:
    """
    Get a stable path to the cutlass folder in the package, with safe failure if it doesn't exist.
    
    Args:
        package_name: Name of the installed Python package.
        folder_path: Relative path to the folder (e.g., 'thirdparty/cutlass').
    
    Returns:
        Path to the folder if it exists, None otherwise.
    """
    package_path = importlib.resources.files('cuda_jit_ptx_test.thirdparty')
        
def thirdparty_dir() -> Path:
    # directory *beside* the package (wheel layout)
    wheel_copy = importlib.resources.files("cuda_jit_ptx_test").parent / "thirdparty"
    # directory *beside* src/ (editable / repo checkout)
    repo_copy  = Path(__file__).resolve().parents[2] / "thirdparty"

    for path in (repo_copy, wheel_copy):
        if path.is_dir():
            return path.resolve()

    raise FileNotFoundError("thirdparty directory not found")

def cuda_include_dir() -> Path:
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

    code = read_local_cuda()

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
                        get_resource_dir('cuda'),
                        thirdparty_dir() / 'cutlass/include',
                        thirdparty_dir() / 'cutlass/tools/util/include',
                        cuda_include_dir()
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