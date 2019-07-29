#ifndef CONVTRANSPOSE_H
#define CONVTRANSPOSE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ConvTranspose : public Layer {
    public:
        ConvTranspose() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~ConvTranspose(){}

    };
}

#endif
