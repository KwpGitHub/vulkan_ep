#ifndef IDENTITY_H
#define IDENTITY_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Identity : public Layer {
    public:
        Identity() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Identity(){}

    };
}

#endif
