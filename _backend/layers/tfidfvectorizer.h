#ifndef TFIDFVECTORIZER_H
#define TFIDFVECTORIZER_H //TfIdfVectorizer

//INPUTS:                   X
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts, ngram_indexes
//PARAMETER_TYPES:          INT, INT, INT, STRING, INTS, INTS
//OPTIONAL_PARAMETERS:      pool_int64s, pool_strings, weights
//OPTIONAL_PARAMETERS_TYPE: INTS, STRINGS, FLOATS

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class TfIdfVectorizer : public Layer {
        
        vuh::Device* _get_device();

        struct Params{ };
        vuh::Program<Specs, Params>* program;

    public:
        TfIdfVectorizer(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
         
         //std::vector<uint32_t> output_shape();
   
        ~TfIdfVectorizer(){}
    };
}


namespace backend {    
    TfIdfVectorizer::TfIdfVectorizer(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/tfidfvectorizer.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* TfIdfVectorizer::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }


};

#endif
