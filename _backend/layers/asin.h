#ifndef ASIN_H
#define ASIN_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Asin : public Layer {
    public:
        Asin() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Asin(){}

    };
}

#endif
