#include "../include/layer_operations.h"
#include "../include/tensor_math.h"

int linear_forward(Tensor* input, Tensor* weights, Tensor* bias, Tensor* outputs) {
    int result = matmul(input, weights, outputs);

    if (result != 0) {
        return result;
    }

    // in this case it's safe to reuse the output since the add goes from left to right so
    // the value we're stepping over was already used. in this way, we don't have to create
    // a new tensor each time we call a foward pass and we avoid the allcoation alltogether
    result = elementwise_add(outputs, bias, outputs);

    if (result != 0) {
        return result;
    }

    return 0;
}
