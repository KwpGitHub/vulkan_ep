#ifndef LPPOOL_H
#define LPPOOL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class LpPool : public Layer {
    public:
        LpPool() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~LpPool(){}

    };
}

#endif
