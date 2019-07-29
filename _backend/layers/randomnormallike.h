#ifndef RANDOMNORMALLIKE_H
#define RANDOMNORMALLIKE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class RandomNormalLike : public Layer {
    public:
        RandomNormalLike() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~RandomNormalLike(){}

    };
}

#endif
