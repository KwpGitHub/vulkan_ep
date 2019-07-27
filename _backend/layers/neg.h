#ifndef NEG_H
#define NEG_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Neg : public Layer {
    public:
        Neg() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Neg(){}

    };
}

#endif
