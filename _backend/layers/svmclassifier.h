#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class SVMClassifier : public Layer {
    public:
        SVMClassifier() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~SVMClassifier(){}

    };
}

#endif
