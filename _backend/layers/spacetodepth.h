#ifndef SPACETODEPTH_H
#define SPACETODEPTH_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class SpaceToDepth : public Layer {
    public:
        SpaceToDepth() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~SpaceToDepth(){}

    };
}

#endif
