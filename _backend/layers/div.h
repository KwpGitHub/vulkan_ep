#ifndef DIV_H
#define DIV_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Div : public Layer {
    public:
        Div() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Div(){}

    };
}

#endif
