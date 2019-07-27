#ifndef RELU_H
#define RELU_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Relu : public Layer {
    public:
        Relu() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Relu(){}

    };
}

#endif
