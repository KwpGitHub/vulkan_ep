#ifndef LPNORMALIZATION_H
#define LPNORMALIZATION_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class LpNormalization : public Layer {
    public:
        LpNormalization() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~LpNormalization(){}

    };
}

#endif
