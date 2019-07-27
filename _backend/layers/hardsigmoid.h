#ifndef HARDSIGMOID_H
#define HARDSIGMOID_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class HardSigmoid : public Layer {
    public:
        HardSigmoid() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~HardSigmoid(){}

    };
}

#endif
