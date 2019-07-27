#ifndef SOFTSIGN_H
#define SOFTSIGN_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Softsign : public Layer {
    public:
        Softsign() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Softsign(){}

    };
}

#endif
