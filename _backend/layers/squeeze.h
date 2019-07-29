#ifndef SQUEEZE_H
#define SQUEEZE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Squeeze : public Layer {
    public:
        Squeeze() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~Squeeze(){}

    };
}

#endif
