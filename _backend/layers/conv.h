#ifndef CONV_H
#define CONV_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Conv : public Layer {
    public:
        Conv() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Conv(){}

    };
}

#endif
