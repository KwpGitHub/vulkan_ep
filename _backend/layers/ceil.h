#ifndef CEIL_H
#define CEIL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Ceil : public Layer {
    public:
        Ceil() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Ceil(){}

    };
}

#endif
