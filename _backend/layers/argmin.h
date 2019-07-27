#ifndef ARGMIN_H
#define ARGMIN_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ArgMin : public Layer {
    public:
        ArgMin() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~ArgMin(){}

    };
}

#endif
