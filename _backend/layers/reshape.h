#ifndef RESHAPE_H
#define RESHAPE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Reshape : public Layer {
    public:
        Reshape() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~Reshape(){}

    };
}

#endif
