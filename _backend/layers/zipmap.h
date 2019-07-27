#ifndef ZIPMAP_H
#define ZIPMAP_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ZipMap : public Layer {
    public:
        ZipMap() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~ZipMap(){}

    };
}

#endif
