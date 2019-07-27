#ifndef ATAN_H
#define ATAN_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Atan : public Layer {
    public:
        Atan() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Atan(){}

    };
}

#endif
