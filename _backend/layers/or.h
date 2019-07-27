#ifndef OR_H
#define OR_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Or : public Layer {
    public:
        Or() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Or(){}

    };
}

#endif
