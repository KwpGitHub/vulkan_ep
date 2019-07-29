#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class LeakyRelu : public Layer {
    public:
        LeakyRelu() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~LeakyRelu(){}

    };
}

#endif
