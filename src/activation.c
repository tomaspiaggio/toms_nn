#include "../include/activation.h"
#include <math.h>

void relu(Tensor *in, Tensor *out) {
    unsigned int length = tensor_data_length(in);
    for (unsigned int i = 0; i < length; i++) {
        if (in->data[i] >= 0) {
            out->data[i] = in->data[i];
        } else {
            out->data[i] = 0;
        }
    }
}

// we're passing the gradient through because if we think of relu, the derivate of x is 1 (just the
// original value) and the derivate of 0 is 0 so we return 0 on the else block.
void relu_backward(Tensor *in, Tensor *grad_out, Tensor *grad_in) {
    unsigned int length = tensor_data_length(in);
    for (unsigned int i = 0; i < length; i++) {
        if (in->data[i] > 0) {
            grad_in->data[i] = grad_out->data[i];  // pass gradient through
        } else {
            grad_in->data[i] = 0;  // kill gradient
        }
    }
}

void softmax(Tensor *in, Tensor *out) {
    unsigned int length = tensor_data_length(in);

    // max value on data
    double max_value = in->data[0];
    for (unsigned int i = 1; i < length; i++) {
        if (in->data[i] > max_value) {
            max_value = in->data[i];
        }
    }

    // accumulated sigma
    double acc_sigmas = 0;

    // sigma in each location (dynamic programming in order not to calculate again)
    double each_sigma[length];

    for (unsigned int i = 0; i < length; i++) {
        // in->data[i] - max_value is an optimization. it gets the same results but the
        // exponential doesn't explode and overflows
        double sigma = exp(in->data[i] - max_value);
        each_sigma[i] = sigma;
        acc_sigmas += sigma;
    }

    for (unsigned int i = 0; i < length; i++) {
        out->data[i] = each_sigma[i] / acc_sigmas;
    }
}
