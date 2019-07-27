#ifndef SOFTPLUS_H
#define SOFTPLUS_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Softplus : public Layer {
    public:
        Softplus() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Softplus(){}

    };
}

#endif
