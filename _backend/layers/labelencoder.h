#ifndef LABELENCODER_H
#define LABELENCODER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class LabelEncoder : public Layer {
    public:
        LabelEncoder() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~LabelEncoder(){}

    };
}

#endif
