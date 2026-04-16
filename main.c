#include "include/loss.h"
#include "include/tensor_math.h"
#include <stdio.h>
#include <stdlib.h>

int test_matmul() {
    unsigned int shape[] = {2, 2};

    Tensor* trivial = create_tensor(2, shape);
    if (!trivial) return 1;

    Tensor* out = create_tensor(2, shape);
    if (!out) {
        delete_tensor(trivial);
        return 1;
    }

    tensor_set(trivial, (unsigned int[]){0, 0}, 1);
    tensor_set(trivial, (unsigned int[]){0, 1}, 2);
    tensor_set(trivial, (unsigned int[]){1, 0}, 3);
    tensor_set(trivial, (unsigned int[]){1, 1}, 4);
    print_tensor(trivial);

    print_tensor(trivial);
    matmul(trivial, trivial, out);
    print_tensor(out);

    delete_tensor(trivial);
    delete_tensor(out);

    return 0;
}

int test_general_math() {
    unsigned int shape_a[] = {2, 2};
    Tensor* a = create_tensor(2, shape_a);
    if (!a) return 1;

    unsigned int shape_b[] = {2, 2};
    Tensor* b = create_tensor(2, shape_b);
    if (!b) {
        delete_tensor(a);
        return 1;
    }

    unsigned int shape_out[] = {2, 2};
    Tensor* out = create_tensor(2, shape_out);
    if (!out) {
        delete_tensor(a);
        delete_tensor(b);
        return 1;
    }

    unsigned int shape_out2[] = {2, 2};
    Tensor* out2 = create_tensor(2, shape_out2);
    if (!out2) {
        delete_tensor(a);
        delete_tensor(b);
        delete_tensor(out);
        return 1;
    }

    tensor_set(a, (unsigned int[]){0, 0}, 0);
    tensor_set(a, (unsigned int[]){0, 1}, 2);
    tensor_set(a, (unsigned int[]){1, 0}, 4);
    tensor_set(a, (unsigned int[]){1, 1}, 3);
    print_tensor(a);

    tensor_set(b, (unsigned int[]){0, 0}, 0);
    tensor_set(b, (unsigned int[]){0, 1}, 2);
    tensor_set(b, (unsigned int[]){1, 0}, 4);
    tensor_set(b, (unsigned int[]){1, 1}, 3);
    print_tensor(b);

    elementwise_add(a, b, out);
    elementwise_mul(a, b, out2);

    print_tensor(out);
    print_tensor(out2);

    scale(out, 3, out2);

    print_tensor(out2);

    double before = tensor_get(out, (unsigned int[]){0, 1});
    transpose(out, out2);
    double after = tensor_get(out2, (unsigned int[]){0, 1});

    print_tensor(out);

    printf("before: %d, after: %d", (int)before, (int)after);

    return 0;
}

int test_cross_entropy_loss() {
    // predictions
    // [[0.7, 0.1, 0.1, 0.1],
    //  [0.1, 0.2, 0.6, 0.1],
    //  [0.2, 0.2, 0.2, 0.4]]
    //
    // targets
    // [[1, 0, 0, 0],
    //  [0, 0, 1, 0],
    //  [0, 0, 0, 1]]

    printf("about to create targets\n");
    Tensor* targets = create_tensor(2, (unsigned int[]){3, 4});
    if (!targets) {
        return 1;
    }

    printf("created targets\n");

    double targets_data[12] = {
        1, 0, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    for (unsigned int i = 0; i < 12; i++) {
        targets->data[i] = targets_data[i];
    }

    Tensor* predictions = create_tensor(2, (unsigned int[]){3, 4});
    if (!predictions) {
        delete_tensor(targets);
        return 1;
    }

    printf("created predictions\n");

    double predictions_data[12] = {
        .7, .1, .1, .1,
        .1, .2, .6, .1,
        .2, .2, .2, .4
    };

    for (unsigned int i = 0; i < 12; i++) {
        predictions->data[i] = predictions_data[i];
    }

    printf("about to calculate cross entropy loss\n");
    double result = cross_entropy_loss(predictions, targets);

    printf("the cross_entropy_loss for the following tensors\n\n");

    printf("predictions\n");
    print_tensor(predictions);

    printf("\n\n");

    printf("targets\n");
    print_tensor(targets);

    printf("\n\nis = %f", result);

    delete_tensor(predictions);
    delete_tensor(targets);

    return 0;
}


int main() {
    int result = 0;

    printf("Testing general math\n");
    result = test_general_math();

    if (result != 0) {
        return result;
    }

    printf("\n\n\n");
    printf("Testing matmul\n");
    result = test_matmul();

    if (result != 0) {
        return result;
    }

    printf("\n\n\n");
    printf("Testing cross_entropy_loss\n");
    result = test_cross_entropy_loss();

    if (result != 0) {
        return result;
    }

    return 0;
}
