#include "tfidfvectorizer.h"
//cpp stuff
namespace layers {    
   
    TfIdfVectorizer::TfIdfVectorizer(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/tfidfvectorizer.spv");       
        dev = backend::g_device;
    }
       
        
    void TfIdfVectorizer::init( int _max_gram_length,  int _max_skip_count,  int _min_gram_length,  std::string _mode,  std::vector<int> _ngram_counts,  std::vector<int> _ngram_indexes,  std::vector<int> _pool_int64s,  std::vector<std::string> _pool_strings,  std::vector<float> _weights) {      
		 m_max_gram_length = _max_gram_length; 
 		 m_max_skip_count = _max_skip_count; 
 		 m_min_gram_length = _min_gram_length; 
 		 m_mode = _mode; 
 		 m_ngram_counts = _ngram_counts; 
 		 m_ngram_indexes = _ngram_indexes; 
 		 m_pool_int64s = _pool_int64s; 
 		 m_pool_strings = _pool_strings; 
 		 m_weights = _weights; 
  

    }
    
    void TfIdfVectorizer::bind(std::string _X_i, std::string _Y_o){    
        m_X_i = _X_i; m_Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void TfIdfVectorizer::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data);
    }

    void TfIdfVectorizer::forward(){ 
        program->run();
    }

}

