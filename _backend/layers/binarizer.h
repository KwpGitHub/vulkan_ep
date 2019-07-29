#ifndef BINARIZER_H
#define BINARIZER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Binarizer : public Layer {
    public:
        Binarizer() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~Binarizer(){}

    };
}

#endif
