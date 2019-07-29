#ifndef ARGMAX_H
#define ARGMAX_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ArgMax : public Layer {
    public:
        ArgMax() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~ArgMax(){}

    };
}

#endif
