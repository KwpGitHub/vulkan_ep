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
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~DepthToSpace(){}

    };
}

#endif
