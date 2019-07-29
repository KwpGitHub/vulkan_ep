#ifndef ONEHOTENCODER_H
#define ONEHOTENCODER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class OneHotEncoder : public Layer {
    public:
        OneHotEncoder() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~OneHotEncoder(){}

    };
}

#endif
