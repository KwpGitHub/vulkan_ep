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
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~FeatureVectorizer(){}

    };
}

#endif
