#ifndef REDUCESUMSQUARE_H
#define REDUCESUMSQUARE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceSumSquare : public Layer {
    public:
        ReduceSumSquare() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~ReduceSumSquare(){}

    };
}

#endif
