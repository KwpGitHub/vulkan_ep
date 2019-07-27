#ifndef REDUCEL1_H
#define REDUCEL1_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceL1 : public Layer {
    public:
        ReduceL1() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~ReduceL1(){}

    };
}

#endif
