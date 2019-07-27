#ifndef RNN_H
#define RNN_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class RNN : public Layer {
    public:
        RNN() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~RNN(){}

    };
}

#endif
