#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

int cross_entropy_loss(Tensor* predictions, Tensor* targets);
int cross_entropy_loss_backward(Tensor* predictions, Tensor* targets, Tensor* grad);

#endif
