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
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~SpaceToDepth(){}

    };
}

#endif
