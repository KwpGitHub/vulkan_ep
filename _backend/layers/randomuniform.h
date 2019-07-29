#ifndef RANDOMUNIFORM_H
#define RANDOMUNIFORM_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class RandomUniform : public Layer {
    public:
        RandomUniform() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~RandomUniform(){}

    };
}

#endif
