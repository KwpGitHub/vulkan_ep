#ifndef REDUCEPROD_H
#define REDUCEPROD_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceProd : public Layer {
    public:
        ReduceProd() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~ReduceProd(){}

    };
}

#endif
