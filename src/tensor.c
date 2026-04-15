#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"

Tensor* create_tensor(int ndim, unsigned int* shape) {
    Tensor* t = malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->ndim = ndim;

    // copy shape in and compute strides
    unsigned int total = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        t->shape[i] = shape[i];
        t->strides[i] = total;
        total *= shape[i];
    }

    // allocate the actual data
    t->data = calloc(total, sizeof(double));
    if (!t->data) {
        free(t);
        return NULL;
    }

    return t;
}

void delete_tensor(Tensor* t) {
    if (!t) return;
    free(t->data);
    free(t);
}

void print_tensor(Tensor* t) {
    unsigned int total = 1;
    for (unsigned int i = 0; i < t->ndim; i++) total *= t->shape[i];

    printf("Tensor(shape=[");
    for (unsigned int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->shape[i], i + 1 < t->ndim ? ", " : "");
    }
    printf("], data=[");
    for (unsigned int i = 0; i < total; i++) {
        printf("%.2f%s", t->data[i], i + 1 < total ? ", " : "");
    }
    printf("])\n");
}

unsigned int tensor_data_length(Tensor *t) {
    unsigned int length = 1;

    for (unsigned int i = 0; i < t->ndim; i++) {
        length *= t->shape[i];
    }

    return length;
}
