#ifndef ARRAYFEATUREEXTRACTOR_H
#define ARRAYFEATUREEXTRACTOR_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ArrayFeatureExtractor : public Layer {
    public:
        ArrayFeatureExtractor() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~ArrayFeatureExtractor(){}

    };
}

#endif
