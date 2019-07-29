#ifndef MULTINOMIAL_H
#define MULTINOMIAL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Multinomial : public Layer {
    public:
        Multinomial() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~Multinomial(){}

    };
}

#endif
