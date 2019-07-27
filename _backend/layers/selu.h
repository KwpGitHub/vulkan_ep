#ifndef SELU_H
#define SELU_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Selu : public Layer {
    public:
        Selu() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Selu(){}

    };
}

#endif
