#ifndef RESHAPE_H
#define RESHAPE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Reshape : public Layer {
    public:
        Reshape() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Reshape(){}

    };
}

#endif
