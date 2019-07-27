#ifndef REDUCESUMSQUARE_H
#define REDUCESUMSQUARE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceSumSquare : public Layer {
    public:
        ReduceSumSquare() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~ReduceSumSquare(){}

    };
}

#endif
