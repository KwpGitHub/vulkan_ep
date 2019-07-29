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
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~SVMRegressor(){}

    };
}

#endif
