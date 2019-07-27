#ifndef SQUEEZE_H
#define SQUEEZE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Squeeze : public Layer {
    public:
        Squeeze() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Squeeze(){}

    };
}

#endif
