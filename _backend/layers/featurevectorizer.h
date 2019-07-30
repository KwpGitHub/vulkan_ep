#ifndef FEATUREVECTORIZER_H
#define FEATUREVECTORIZER_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class FeatureVectorizer : public Layer {
    public:
        FeatureVectorizer(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {}
        
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
