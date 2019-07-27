#ifndef RECIPROCAL_H
#define RECIPROCAL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Reciprocal : public Layer {
    public:
        Reciprocal() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Reciprocal(){}

    };
}

#endif
