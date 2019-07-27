#ifndef GLOBALAVERAGEPOOL_H
#define GLOBALAVERAGEPOOL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class GlobalAveragePool : public Layer {
    public:
        GlobalAveragePool() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~GlobalAveragePool(){}

    };
}

#endif
