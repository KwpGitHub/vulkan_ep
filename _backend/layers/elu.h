#ifndef ELU_H
#define ELU_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Elu : public Layer {
    public:
        Elu() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Elu(){}

    };
}

#endif
