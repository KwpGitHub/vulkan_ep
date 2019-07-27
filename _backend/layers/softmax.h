#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Softmax : public Layer {
    public:
        Softmax() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Softmax(){}

    };
}

#endif
