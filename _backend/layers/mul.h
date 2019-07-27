#ifndef MUL_H
#define MUL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Mul : public Layer {
    public:
        Mul() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Mul(){}

    };
}

#endif
