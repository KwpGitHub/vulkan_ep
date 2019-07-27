#ifndef SIGMOID_H
#define SIGMOID_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Sigmoid : public Layer {
    public:
        Sigmoid() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Sigmoid(){}

    };
}

#endif
