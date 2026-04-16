#include "tensor.h"

#ifndef LAYER_OPERATIONS_H
#define LAYER_OPERATIONS_H

int linear_forward(Tensor* input, Tensor* weights, Tensor* bias, Tensor* outputs);
int linear_backward();

#endif
