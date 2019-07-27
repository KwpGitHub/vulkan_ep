#ifndef CASTMAP_H
#define CASTMAP_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class CastMap : public Layer {
    public:
        CastMap() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~CastMap(){}

    };
}

#endif
