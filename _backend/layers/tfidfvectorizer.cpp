#include "TfIdfVectorizer.h"
//cpp stuff
namespace backend {    
   
    TfIdfVectorizer::TfIdfVectorizer(std::string name) : Layer(name) { }
       
    vuh::Device* TfIdfVectorizer::_get_device() {
        
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
    
    void TfIdfVectorizer::bind(std::string _pool_strings, std::string _weights, std::string _X_i, std::string _Y_o){
        pool_strings = _pool_strings; weights = _weights; X_i = _X_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
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
}

