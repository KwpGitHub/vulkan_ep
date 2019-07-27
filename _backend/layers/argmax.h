#ifndef ARGMAX_H
#define ARGMAX_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ArgMax : public Layer {
    public:
        ArgMax() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~ArgMax(){}

    };
}

#endif
