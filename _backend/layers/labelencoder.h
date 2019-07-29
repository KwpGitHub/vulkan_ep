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
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~LabelEncoder(){}

    };
}

#endif
