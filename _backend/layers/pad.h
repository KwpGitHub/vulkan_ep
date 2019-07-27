#ifndef PAD_H
#define PAD_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Pad : public Layer {
    public:
        Pad() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Pad(){}

    };
}

#endif
