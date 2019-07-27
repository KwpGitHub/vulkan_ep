#ifndef MAXROIPOOL_H
#define MAXROIPOOL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class MaxRoiPool : public Layer {
    public:
        MaxRoiPool() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~MaxRoiPool(){}

    };
}

#endif
