#ifndef RECIPROCAL_H
#define RECIPROCAL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Reciprocal : public Layer {
    public:
        Reciprocal() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~Reciprocal(){}

    };
}

#endif
