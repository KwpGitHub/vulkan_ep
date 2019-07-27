#ifndef CLIP_H
#define CLIP_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Clip : public Layer {
    public:
        Clip() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Clip(){}

    };
}

#endif
