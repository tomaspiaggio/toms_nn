#include "../include/tensor_math.h"
#include <stdio.h>

// TODO: in general i think i'm not copying the data correctly. or not checking. i need to decide one way or
// the other. either checking or copying. like for example on transpose. the tensor that comes in could have
// incorrect strides and whatnot. i should either check or fix the values. i think checking is more polite.

int matmul(Tensor *A, Tensor *B, Tensor *out) {
    if (A->ndim != B->ndim || A->ndim < 2) {
        printf("ndims don't match or are too small\n");
        return -1;
    }

    // batch dims must match exactly
    for (unsigned int i = 0; i < A->ndim - 2; i++) {
        if (A->shape[i] != B->shape[i]) {
            printf("batch dims must match exactly (except for the last two) -> A->shape[i] != B->shape[i] for i = %d. A: %d, B: %d\n", i, A->shape[i], B->shape[i]);
            return -1;
        }
    }

    // inner dims must match: A's last == B's second to last
    int ndim = (int)A->ndim;
    if (A->shape[ndim - 1] != B->shape[ndim - 2]) {
        printf("the last dimention of A must match the second to last dimention of B. A->shape[ndim - 1] != B->shape[ndim - 2] -> A: %d, B: %d\n", A->shape[ndim - 1], B->shape[ndim - 2]);
        return -1;
    }

    unsigned int M = A->shape[ndim - 2];  // rows of each matrix
    unsigned int K = A->shape[ndim - 1];  // shared inner dim
    unsigned int N = B->shape[ndim - 1];  // cols of each matrix
    unsigned int n_batches = tensor_data_length(A);

    for (unsigned int batch = 0; batch < n_batches; batch++) {
        // convert flat batch index to batch offsets for A, B, out
        unsigned int a_offset = 0;
        unsigned int b_offset = 0;
        unsigned int o_offset = 0;
        unsigned int remaining = batch;

        for (int d = ndim - 3; d >= 0; d--) {
            unsigned int idx = remaining % A->shape[d];
            remaining /= A->shape[d];
            a_offset += idx * A->strides[d];
            b_offset += idx * B->strides[d];
            o_offset += idx * out->strides[d];
        }

        // standard 2D matmul at these offsets
        for (unsigned int i = 0; i < M; i++) {
            for (unsigned int j = 0; j < N; j++) {
                double sum = 0;
                for (unsigned int k = 0; k < K; k++) {
                    double a_val = A->data[a_offset + i * A->strides[ndim-2] + k * A->strides[ndim-1]];
                    double b_val = B->data[b_offset + k * B->strides[ndim-2] + j * B->strides[ndim-1]];
                    sum += a_val * b_val;
                }
                out->data[o_offset + i * out->strides[ndim-2] + j * out->strides[ndim-1]] = sum;
            }
        }
    }

    return 0;
}

int elementwise_add(Tensor *A, Tensor *B, Tensor *out) {
    if (A->ndim != B->ndim) {
        printf("dimentions for A and B are different: %d != %d\n", A->ndim, B->ndim);
        return -1;
    }

    for (unsigned int i = 0; i < A->ndim; i++) {
        if (A->shape[i] != B->shape[i]) {
            printf("strides for A and B are different\n");
            return -1;
        }
    }

    unsigned int length = tensor_data_length(A);
    for (unsigned int i = 0; i < length; i++) {
        out->data[i] = A->data[i] + B->data[i];
    }

    return 0;
}

int elementwise_mul(Tensor *A, Tensor *B, Tensor *out) {
    if (A->ndim != B->ndim) {
        printf("dimentions for A and B are different: %d != %d\n", A->ndim, B->ndim);
        return -1;
    }

    for (unsigned int i = 0; i < A->ndim; i++) {
        if (A->shape[i] != B->shape[i]) {
            printf("strides for A and B are different\n");
            return -1;
        }
    }

    unsigned int length = tensor_data_length(A);
    for (unsigned int i = 0; i < length; i++) {
        out->data[i] = A->data[i] * B->data[i];
    }

    return 0;
}

int transpose(Tensor* A, Tensor* out) {
    // we will copy the data because the data of tensor A could be freed and we have no control over that

    for (unsigned int i = 0; i < A->ndim; i++) {
        out->strides[i] = A->strides[A->ndim - i - 1];
        out->shape[i] = A->shape[A->ndim - i - 1];
    }

    unsigned int length = tensor_data_length(A);
    for (unsigned int i = 0; i < length; i++) {
        out->data[i] = A->data[i];
    }

    out->ndim = A->ndim;

    return 0;
}

void scale(Tensor* A, double scalar, Tensor *out) {
    unsigned int length = tensor_data_length(A);

    for (unsigned int i = 0; i < length; i++) {
        out->data[i] = A->data[i] * scalar;
    }
}
