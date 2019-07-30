#ifndef RANDOMUNIFORMLIKE_H
#define RANDOMUNIFORMLIKE_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class RandomUniformLike : public Layer {
    public:
        RandomUniformLike(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {}
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~RandomUniformLike(){}

    };
}

#endif
