#ifndef HARDSIGMOID_H
#define HARDSIGMOID_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class HardSigmoid : public Layer {
    public:
        HardSigmoid() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~HardSigmoid(){}

    };
}

#endif
