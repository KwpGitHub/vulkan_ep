#ifndef XOR_H
#define XOR_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Xor : public Layer {
    public:
        Xor() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Xor(){}

    };
}

#endif
