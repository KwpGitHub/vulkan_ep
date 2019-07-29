#ifndef SOFTSIGN_H
#define SOFTSIGN_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Softsign : public Layer {
    public:
        Softsign() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~Softsign(){}

    };
}

#endif
