#ifndef DICTVECTORIZER_H
#define DICTVECTORIZER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class DictVectorizer : public Layer {
    public:
        DictVectorizer() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~DictVectorizer(){}

    };
}

#endif