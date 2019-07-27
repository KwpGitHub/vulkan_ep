#ifndef CATEGORYMAPPER_H
#define CATEGORYMAPPER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class CategoryMapper : public Layer {
    public:
        CategoryMapper() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~CategoryMapper(){}

    };
}

#endif
