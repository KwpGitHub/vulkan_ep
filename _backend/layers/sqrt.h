#ifndef SQRT_H
#define SQRT_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Sqrt : public Layer {
    public:
        Sqrt() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Sqrt(){}

    };
}

#endif
