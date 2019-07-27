#ifndef RANDOMNORMALLIKE_H
#define RANDOMNORMALLIKE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class RandomNormalLike : public Layer {
    public:
        RandomNormalLike() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~RandomNormalLike(){}

    };
}

#endif
