#ifndef RANDOMNORMAL_H
#define RANDOMNORMAL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class RandomNormal : public Layer {
    public:
        RandomNormal() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~RandomNormal(){}

    };
}

#endif
