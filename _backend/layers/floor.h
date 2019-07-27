#ifndef FLOOR_H
#define FLOOR_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Floor : public Layer {
    public:
        Floor() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Floor(){}

    };
}

#endif
