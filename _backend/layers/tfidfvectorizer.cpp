#include "tfidfvectorizer.h"
//cpp stuff
namespace layers {    
   
    TfIdfVectorizer::TfIdfVectorizer(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/tfidfvectorizer.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* TfIdfVectorizer::_get_device() {        
        return backend::device;
    }
    
    void TfIdfVectorizer::init( int _max_gram_length,  int _max_skip_count,  int _min_gram_length,  std::string _mode,  std::vector<int> _ngram_counts,  std::vector<int> _ngram_indexes,  std::vector<int> _pool_int64s,  std::vector<std::string> _pool_strings,  std::vector<float> _weights) {      
		 max_gram_length = _max_gram_length; 
 		 max_skip_count = _max_skip_count; 
 		 min_gram_length = _min_gram_length; 
 		 mode = _mode; 
 		 ngram_counts = _ngram_counts; 
 		 ngram_indexes = _ngram_indexes; 
 		 pool_int64s = _pool_int64s; 
 		 pool_strings = _pool_strings; 
 		 weights = _weights; 
  
    }
    
    void TfIdfVectorizer::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.max_gram_length = max_gram_length;
  		//binding.max_skip_count = max_skip_count;
  		//binding.min_gram_length = min_gram_length;
  		//binding.mode = mode;
  		//binding.ngram_counts = ngram_counts;
  		//binding.ngram_indexes = ngram_indexes;
  		//binding.pool_int64s = pool_int64s;
  		//binding.pool_strings = pool_strings;
  		//binding.weights = weights;
         
    }

    void TfIdfVectorizer::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void TfIdfVectorizer::forward(){ 
        program->run();
    }

}

