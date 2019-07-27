#ifndef RANDOMUNIFORMLIKE_H
#define RANDOMUNIFORMLIKE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class RandomUniformLike : public Layer {
    public:
        RandomUniformLike() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~RandomUniformLike(){}

    };
}

#endif
