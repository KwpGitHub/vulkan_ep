#ifndef TANH_H
#define TANH_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Tanh : public Layer {
    public:
        Tanh() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Tanh(){}

    };
}

#endif
