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
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~SVMClassifier(){}

    };
}

#endif
