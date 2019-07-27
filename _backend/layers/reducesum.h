#ifndef REDUCESUM_H
#define REDUCESUM_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceSum : public Layer {
    public:
        ReduceSum() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~ReduceSum(){}

    };
}

#endif
