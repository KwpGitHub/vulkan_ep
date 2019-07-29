#ifndef REDUCEL2_H
#define REDUCEL2_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceL2 : public Layer {
    public:
        ReduceL2() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~ReduceL2(){}

    };
}

#endif
