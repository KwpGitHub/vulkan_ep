#ifndef NORMALIZER_H
#define NORMALIZER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Normalizer : public Layer {
    public:
        Normalizer() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~Normalizer(){}

    };
}

#endif
