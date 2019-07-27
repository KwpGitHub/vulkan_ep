#ifndef TAN_H
#define TAN_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Tan : public Layer {
    public:
        Tan() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Tan(){}

    };
}

#endif
