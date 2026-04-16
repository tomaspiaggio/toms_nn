#define MAX_DIMS 8

#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    unsigned int ndim;
    unsigned int shape[MAX_DIMS];
    unsigned int strides[MAX_DIMS];
    double* data;
} Tensor;

// To access element [i, j, k] you compute data[i * strides[0] + j * strides[1] + k * strides[2]].
// This generalizes your row * cols + col pattern to any number of dimensions.
// this has worse performance than just doing the regular access from the other function since we're
// passing stuff around. it's better to jsut do the access unless explicitly needed.
// we can use this function and then replace it later.
static inline double tensor_get(Tensor* t, unsigned int* indices) {
    unsigned int offset = 0;
    for (unsigned int i = 0; i < t->ndim; i++) {
        offset += indices[i] * t->strides[i];
    }
    return t->data[offset];
}

static inline void tensor_set(Tensor* t, unsigned int* indeces, double data) {
    unsigned int offset = 0;
    for (unsigned int i = 0; i < t->ndim; i++) {
        offset += indeces[i] * t->strides[i];
    }

    t->data[offset] = data;
}

Tensor* create_tensor(int ndim, unsigned int* shape);
void delete_tensor(Tensor* t);
void print_tensor(Tensor* t);
unsigned int tensor_data_length(Tensor* t);

#endif
