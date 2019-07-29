#ifndef IMPUTER_H
#define IMPUTER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Imputer : public Layer {
    public:
        Imputer() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~Imputer(){}

    };
}

#endif
