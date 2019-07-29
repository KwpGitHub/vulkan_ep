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
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~LinearClassifier(){}

    };
}

#endif
