#include "TfIdfVectorizer.h"

//cpp stuff
namespace backend {    
   
    TfIdfVectorizer::TfIdfVectorizer(std::string n) : Layer(n) { }
       
    vuh::Device* TfIdfVectorizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void TfIdfVectorizer::init( int _max_gram_length,  int _max_skip_count,  int _min_gram_length,  int _mode,  Shape_t _ngram_counts,  Shape_t _ngram_indexes,  Shape_t _pool_int64s) {      
		 max_gram_length = _max_gram_length; 
 		 max_skip_count = _max_skip_count; 
 		 min_gram_length = _min_gram_length; 
 		 mode = _mode; 
 		 ngram_counts = _ngram_counts; 
 		 ngram_indexes = _ngram_indexes; 
 		 pool_int64s = _pool_int64s; 
  
    }
    
    void TfIdfVectorizer::bind(std::string _pool_strings, std::string _weights, std::string _X_input, std::string _Y_output){
        pool_strings = _pool_strings; weights = _weights; X_input = _X_input; Y_output = _Y_output;
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
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/tfidfvectorizer.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[pool_strings]->data(), *tensor_dict[weights]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    //backend::nn;

//python stuff


