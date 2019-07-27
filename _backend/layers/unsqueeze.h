#ifndef UNSQUEEZE_H
#define UNSQUEEZE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Unsqueeze : public Layer {
    public:
        Unsqueeze() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Unsqueeze(){}

    };
}

#endif
