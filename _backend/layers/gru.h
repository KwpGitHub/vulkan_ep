#ifndef GRU_H
#define GRU_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class GRU : public Layer {
    public:
        GRU() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~GRU(){}

    };
}

#endif
