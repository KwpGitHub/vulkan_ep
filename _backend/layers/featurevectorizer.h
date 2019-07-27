#ifndef FEATUREVECTORIZER_H
#define FEATUREVECTORIZER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class FeatureVectorizer : public Layer {
    public:
        FeatureVectorizer() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~FeatureVectorizer(){}

    };
}

#endif
