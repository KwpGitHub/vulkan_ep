#ifndef REDUCEL2_H
#define REDUCEL2_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceL2 : public Layer {
    public:
        ReduceL2() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~ReduceL2(){}

    };
}

#endif
