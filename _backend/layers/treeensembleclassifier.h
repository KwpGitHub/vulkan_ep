#ifndef TREEENSEMBLECLASSIFIER_H
#define TREEENSEMBLECLASSIFIER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class TreeEnsembleClassifier : public Layer {
    public:
        TreeEnsembleClassifier() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~TreeEnsembleClassifier(){}

    };
}

#endif
