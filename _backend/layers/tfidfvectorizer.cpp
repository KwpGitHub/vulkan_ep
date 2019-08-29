#include "tfidfvectorizer.h"
//cpp stuff
namespace layers {    
   
    TfIdfVectorizer::TfIdfVectorizer(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/tfidfvectorizer.spv");       
        dev = backend::device;
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
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void TfIdfVectorizer::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[X_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[X_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[X_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void TfIdfVectorizer::forward(){ 
        program->run();
    }

}

