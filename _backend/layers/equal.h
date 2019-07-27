#ifndef EQUAL_H
#define EQUAL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Equal : public Layer {
    public:
        Equal() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Equal(){}

    };
}

#endif
