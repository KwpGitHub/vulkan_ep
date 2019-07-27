#ifndef SIN_H
#define SIN_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Sin : public Layer {
    public:
        Sin() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Sin(){}

    };
}

#endif
