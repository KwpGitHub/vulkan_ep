#ifndef MULTINOMIAL_H
#define MULTINOMIAL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Multinomial : public Layer {
    public:
        Multinomial() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Multinomial(){}

    };
}

#endif
