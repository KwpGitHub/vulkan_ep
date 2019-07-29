#ifndef REDUCELOGSUM_H
#define REDUCELOGSUM_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceLogSum : public Layer {
    public:
        ReduceLogSum() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~ReduceLogSum(){}

    };
}

#endif
