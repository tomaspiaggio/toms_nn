#include "include/tensor_math.h"
#include <stdio.h>

int main() {
    // unsigned int shape[] = {3, 4};
    // Tensor* t = create_tensor(2, shape);

    unsigned int shape_a[] = {2, 2};
    Tensor* a = create_tensor(2, shape_a);
    if (!a) return 1;

    unsigned int shape_b[] = {2, 2};
    Tensor* b = create_tensor(2, shape_b);
    if (!b) return 1;

    unsigned int shape_out[] = {2, 2};
    Tensor* out = create_tensor(2, shape_out);
    if (!out) return 1;

    unsigned int shape_out2[] = {2, 2};
    Tensor* out2 = create_tensor(2, shape_out2);
    if (!out2) return 1;

    Tensor* trivial = create_tensor(2, shape_out2);
    if (!trivial) return 1;

    tensor_set(trivial, (unsigned int[]){0, 0}, 1);
    tensor_set(trivial, (unsigned int[]){0, 1}, 2);
    tensor_set(trivial, (unsigned int[]){1, 0}, 3);
    tensor_set(trivial, (unsigned int[]){1, 1}, 4);
    print_tensor(trivial);

    // tensor_set(a, (unsigned int[]){0, 0}, 0);
    // tensor_set(a, (unsigned int[]){0, 1}, 2);
    // tensor_set(a, (unsigned int[]){1, 0}, 4);
    // tensor_set(a, (unsigned int[]){1, 1}, 3);
    // print_tensor(a);

    // tensor_set(b, (unsigned int[]){0, 0}, 0);
    // tensor_set(b, (unsigned int[]){0, 1}, 2);
    // tensor_set(b, (unsigned int[]){1, 0}, 4);
    // tensor_set(b, (unsigned int[]){1, 1}, 3);
    // print_tensor(b);

    // elementwise_add(a, b, out);
    // elementwise_mul(a, b, out2);

    // print_tensor(out);
    // print_tensor(out2);

    // scale(out, 3, out2);

    // print_tensor(out2);

    // double before = tensor_get(out, (unsigned int[]){0, 1});
    // transpose(out, out2);
    // double after = tensor_get(out2, (unsigned int[]){0, 1});

    // print_tensor(out);

    // printf("before: %d, after: %d", (int)before, (int)after);

    print_tensor(trivial);
    matmul(trivial, trivial, out);
    print_tensor(out);


    return 0;
}
