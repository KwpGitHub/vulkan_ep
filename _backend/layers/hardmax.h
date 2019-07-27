#ifndef HARDMAX_H
#define HARDMAX_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Hardmax : public Layer {
    public:
        Hardmax() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Hardmax(){}

    };
}

#endif
