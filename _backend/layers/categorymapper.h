#ifndef CATEGORYMAPPER_H
#define CATEGORYMAPPER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class CategoryMapper : public Layer {
    public:
        CategoryMapper() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~CategoryMapper(){}

    };
}

#endif
