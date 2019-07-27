#ifndef GLOBALLPPOOL_H
#define GLOBALLPPOOL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class GlobalLpPool : public Layer {
    public:
        GlobalLpPool() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~GlobalLpPool(){}

    };
}

#endif
