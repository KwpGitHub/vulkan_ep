#ifndef GLOBALMAXPOOL_H
#define GLOBALMAXPOOL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class GlobalMaxPool : public Layer {
    public:
        GlobalMaxPool() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~GlobalMaxPool(){}

    };
}

#endif
