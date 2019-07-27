#ifndef LRN_H
#define LRN_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class LRN : public Layer {
    public:
        LRN() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~LRN(){}

    };
}

#endif
