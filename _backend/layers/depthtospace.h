#ifndef DEPTHTOSPACE_H
#define DEPTHTOSPACE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class DepthToSpace : public Layer {
    public:
        DepthToSpace() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~DepthToSpace(){}

    };
}

#endif
