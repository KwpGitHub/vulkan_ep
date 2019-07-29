#ifndef UNSQUEEZE_H
#define UNSQUEEZE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Unsqueeze : public Layer {
    public:
        Unsqueeze() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~Unsqueeze(){}

    };
}

#endif
