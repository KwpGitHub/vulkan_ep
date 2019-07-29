#ifndef RANDOMUNIFORMLIKE_H
#define RANDOMUNIFORMLIKE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class RandomUniformLike : public Layer {
    public:
        RandomUniformLike() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~RandomUniformLike(){}

    };
}

#endif
