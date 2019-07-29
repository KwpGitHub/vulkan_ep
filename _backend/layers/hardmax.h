#ifndef HARDMAX_H
#define HARDMAX_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Hardmax : public Layer {
    public:
        Hardmax() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~Hardmax(){}

    };
}

#endif
