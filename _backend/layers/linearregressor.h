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
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~LinearRegressor(){}

    };
}

#endif
