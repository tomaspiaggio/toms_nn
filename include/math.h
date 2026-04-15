#include "tensor.h"

// all these functions need to be immutable for easier debugging
int matmul(Tensor* A, Tensor* B, Tensor* out);
int elementwise_add(Tensor* A, Tensor* B, Tensor* out);
int elementwise_mul(Tensor* A, Tensor* B, Tensor* out);
int transpose(Tensor* A, Tensor* out);
void scale(Tensor* A, double scalar, Tensor* out);
