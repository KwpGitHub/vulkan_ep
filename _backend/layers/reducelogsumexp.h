#ifndef REDUCELOGSUMEXP_H
#define REDUCELOGSUMEXP_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceLogSumExp : public Layer {
    public:
        ReduceLogSumExp() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~ReduceLogSumExp(){}

    };
}

#endif
