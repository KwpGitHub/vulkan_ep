#ifndef SVMREGRESSOR_H
#define SVMREGRESSOR_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class SVMRegressor : public Layer {
    public:
        SVMRegressor() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~SVMRegressor(){}

    };
}

#endif
