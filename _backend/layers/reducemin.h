#ifndef REDUCEMIN_H
#define REDUCEMIN_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceMin : public Layer {
    public:
        ReduceMin() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~ReduceMin(){}

    };
}

#endif
