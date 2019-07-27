#ifndef SPLIT_H
#define SPLIT_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Split : public Layer {
    public:
        Split() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Split(){}

    };
}

#endif
