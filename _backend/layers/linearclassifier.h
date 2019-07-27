#ifndef LINEARCLASSIFIER_H
#define LINEARCLASSIFIER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class LinearClassifier : public Layer {
    public:
        LinearClassifier() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~LinearClassifier(){}

    };
}

#endif
