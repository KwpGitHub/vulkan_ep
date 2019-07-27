#ifndef NOT_H
#define NOT_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Not : public Layer {
    public:
        Not() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Not(){}

    };
}

#endif
