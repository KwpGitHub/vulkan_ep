#ifndef SIGMOID_H
#define SIGMOID_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Sigmoid : public Layer {
    public:
        Sigmoid() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~Sigmoid(){}

    };
}

#endif
