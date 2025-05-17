// #include <stdio.h>
#include <header_test.cuh>
#include <cute/tensor.hpp>

extern "C"
__global__
void
kernel() {
    using namespace cute;

    printf("hello from kernel: %d\n", func());

    auto s = make_layout(make_shape(_4{},_4{}));
    auto i = make_identity_layout(shape(s));
    print("s : "); print(s); print("\n");
    print("i : "); print(i); print("\n");
}