#ifndef REDUCELOGSUMEXP_H
#define REDUCELOGSUMEXP_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceLogSumExp : public Layer {
    public:
        ReduceLogSumExp() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~ReduceLogSumExp(){}

    };
}

#endif
