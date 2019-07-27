#ifndef RANDOMNORMAL_H
#define RANDOMNORMAL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class RandomNormal : public Layer {
    public:
        RandomNormal() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~RandomNormal(){}

    };
}

#endif
