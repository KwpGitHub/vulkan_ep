#ifndef REDUCELOGSUM_H
#define REDUCELOGSUM_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceLogSum : public Layer {
    public:
        ReduceLogSum() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~ReduceLogSum(){}

    };
}

#endif
