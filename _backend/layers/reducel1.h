#ifndef REDUCEL1_H
#define REDUCEL1_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceL1 : public Layer {
    public:
        ReduceL1() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~ReduceL1(){}

    };
}

#endif
