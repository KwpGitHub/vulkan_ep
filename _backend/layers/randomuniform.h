#ifndef RANDOMUNIFORM_H
#define RANDOMUNIFORM_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class RandomUniform : public Layer {
    public:
        RandomUniform() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~RandomUniform(){}

    };
}

#endif
