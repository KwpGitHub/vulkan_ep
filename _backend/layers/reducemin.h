#ifndef REDUCEMIN_H
#define REDUCEMIN_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceMin : public Layer {
    public:
        ReduceMin() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~ReduceMin(){}

    };
}

#endif
