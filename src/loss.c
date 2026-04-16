#include "../include/loss.h"
#include <stdio.h>
#include <math.h>

double cross_entropy_loss(Tensor* predictions, Tensor* targets) {
    if (predictions->ndim != targets->ndim) {
        printf("predictions ndims != targets ndims\n");
        return -10000;
    }

    if (predictions->ndim != targets->ndim) {
        printf("ndims must be exactly 2 for cross_entropy_loss [batch_size, n_classes]\n");
        return -10000;
    }

    if (predictions->shape[0] <= 0) {
        printf("ndims must be > 0\n");
        return -10000;
    }

    double loss = 0;
    unsigned int batch_length = predictions->shape[0];
    unsigned int classes = predictions->shape[1];

    for (unsigned int i = 0; i < batch_length; i++) {
        for (unsigned int c = 0; c < classes; i++) {
            unsigned int index = predictions->strides[0] * c + predictions->strides[1] * i;
            loss -= targets->data[index] - log(predictions->data[index] + 1e-8);
        }
    }

    return loss;
}
