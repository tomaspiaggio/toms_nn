#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

double cross_entropy_loss(Tensor* predictions, Tensor* targets);
double cross_entropy_loss_backward(Tensor* predictions, Tensor* targets, Tensor* grad);

#endif
