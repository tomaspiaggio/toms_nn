#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "./tensor.h"

void relu(Tensor* in, Tensor* out);
// all the "backwards" are the derivates
void relu_backward(Tensor* in, Tensor* grad_out, Tensor* grad_in);
void softmax(Tensor* in, Tensor* out);
// we're not implementing softmax_backward because it's almost never used by itself, but
// in combination with cross_entropy_loss, which in turn, also makes it easier to implement

#endif
