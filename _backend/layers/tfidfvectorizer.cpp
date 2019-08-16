#include "TfIdfVectorizer.h"

//cpp stuff
namespace backend {    
   
    TfIdfVectorizer::TfIdfVectorizer(std::string n, int max_gram_length, int max_skip_count, int min_gram_length, int mode, Shape_t ngram_counts, Shape_t ngram_indexes, Shape_t pool_int64s) : Layer(n) { }
       
    vuh::Device* TfIdfVectorizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void TfIdfVectorizer::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.max_gram_length = max_gram_length;
  		binding.max_skip_count = max_skip_count;
  		binding.min_gram_length = min_gram_length;
  		binding.mode = mode;
  		binding.ngram_counts = ngram_counts;
  		binding.ngram_indexes = ngram_indexes;
  		binding.pool_int64s = pool_int64s;
  		binding.pool_strings = tensor_dict[pool_strings]->shape();
  		binding.weights = tensor_dict[weights]->shape();
 
    }
    
    void TfIdfVectorizer::call(std::string pool_strings, std::string weights, std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/tfidfvectorizer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[pool_strings]->data(), *tensor_dict[weights]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


