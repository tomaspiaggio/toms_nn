#include "tensor.h"

#ifndef LAYER_OPERATIONS_H
#define LAYER_OPERATIONS_H

int linear_forward(Tensor* input, Tensor* weights, Tensor* bias, Tensor* outputs, Tensor* batch, Tensor* in_dim, Tensor* out_dim);

#endif
