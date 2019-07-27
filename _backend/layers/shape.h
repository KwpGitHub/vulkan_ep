#ifndef SHAPE_H
#define SHAPE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Shape : public Layer {
    public:
        Shape() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Shape(){}

    };
}

#endif
