#ifndef EXP_H
#define EXP_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Exp : public Layer {
    public:
        Exp() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Exp(){}

    };
}

#endif
