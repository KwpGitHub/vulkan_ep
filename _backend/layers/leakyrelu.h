#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class LeakyRelu : public Layer {
    public:
        LeakyRelu() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~LeakyRelu(){}

    };
}

#endif
