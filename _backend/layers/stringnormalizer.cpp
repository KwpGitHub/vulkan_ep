#include "stringnormalizer.h"
//cpp stuff
namespace layers {    
   
    StringNormalizer::StringNormalizer(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/stringnormalizer.spv");       
        dev = backend::g_device;
    }
       
        
    void StringNormalizer::init( std::string _case_change_action,  int _is_case_sensitive,  std::string _locale,  std::vector<std::string> _stopwords) {      
		 m_case_change_action = _case_change_action; 
 		 m_is_case_sensitive = _is_case_sensitive; 
 		 m_locale = _locale; 
 		 m_stopwords = _stopwords; 
  

    }
    
    void StringNormalizer::bind(std::string _X_i, std::string _Y_o){    
        m_X_i = _X_i; m_Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void StringNormalizer::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data);
    }

    void StringNormalizer::forward(){ 
        program->run();
    }

}

