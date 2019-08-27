#include "stringnormalizer.h"
//cpp stuff
namespace layers {    
   
    StringNormalizer::StringNormalizer(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/stringnormalizer.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* StringNormalizer::_get_device() {        
        return backend::device;
    }
    
    void StringNormalizer::init( std::string _case_change_action,  int _is_case_sensitive,  std::string _locale,  std::vector<std::string> _stopwords) {      
		 case_change_action = _case_change_action; 
 		 is_case_sensitive = _is_case_sensitive; 
 		 locale = _locale; 
 		 stopwords = _stopwords; 
  
    }
    
    void StringNormalizer::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.case_change_action = case_change_action;
  		//binding.is_case_sensitive = is_case_sensitive;
  		//binding.locale = locale;
  		//binding.stopwords = stopwords;
         
    }

    void StringNormalizer::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void StringNormalizer::forward(){ 
        program->run();
    }

}

