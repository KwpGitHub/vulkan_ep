#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Transpose : public Layer {
    public:
        Transpose() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~Transpose(){}

    };
}

#endif
