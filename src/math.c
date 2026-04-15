#include <stdio.h>
#include "../include/tensor.h"

// TODO: in general i think i'm not copying the data correctly. or not checking. i need to decide one way or
// the other. either checking or copying. like for example on transpose. the tensor that comes in could have
// incorrect strides and whatnot. i should either check or fix the values. i think checking is more polite.

int matmul(Tensor *A, Tensor *B, Tensor *out) {
    // TODO: implement this
    return -1;
}

int elementwise_add(Tensor *A, Tensor *B, Tensor *out) {
    if (A->ndim != B->ndim) {
        printf("dimentions for A and B are different: %d != %d", A->ndim, B->ndim);
        return -1;
    }

    for (unsigned int i = 0; i < A->ndim; i++) {
        if (A->shape[i] != B->shape[i]) {
            printf("strides for A and B are different");
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
        printf("dimentions for A and B are different: %d != %d", A->ndim, B->ndim);
        return -1;
    }

    for (unsigned int i = 0; i < A->ndim; i++) {
        if (A->shape[i] != B->shape[i]) {
            printf("strides for A and B are different");
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
