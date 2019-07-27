#ifndef SIZE_H
#define SIZE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Size : public Layer {
    public:
        Size() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Size(){}

    };
}

#endif
