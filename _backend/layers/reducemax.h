#ifndef REDUCEMAX_H
#define REDUCEMAX_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceMax : public Layer {
    public:
        ReduceMax() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~ReduceMax(){}

    };
}

#endif
