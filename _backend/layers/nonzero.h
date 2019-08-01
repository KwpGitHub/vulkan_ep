#ifndef NONZERO_H
#define NONZERO_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class NonZero : public Layer {
    public:
        NonZero(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {}
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~NonZero(){}

    };
}

#endif
