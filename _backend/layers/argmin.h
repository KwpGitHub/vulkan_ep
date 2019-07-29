#ifndef ARGMIN_H
#define ARGMIN_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ArgMin : public Layer {
    public:
        ArgMin() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~ArgMin(){}

    };
}

#endif
