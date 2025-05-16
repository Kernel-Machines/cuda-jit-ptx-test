
import sys

from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch, ObjectCode

code = r"""

extern "C"
__global__ 
void 
hello() {
    printf("hello from device\n");
}

"""

def run():

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