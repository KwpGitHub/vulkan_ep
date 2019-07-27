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
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~DictVectorizer(){}

    };
}

#endif
