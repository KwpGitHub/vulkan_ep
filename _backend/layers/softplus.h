#ifndef SOFTPLUS_H
#define SOFTPLUS_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Softplus : public Layer {
    public:
        Softplus() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~Softplus(){}

    };
}

#endif
