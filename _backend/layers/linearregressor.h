#ifndef LINEARREGRESSOR_H
#define LINEARREGRESSOR_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class LinearRegressor : public Layer {
    public:
        LinearRegressor() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~LinearRegressor(){}

    };
}

#endif
