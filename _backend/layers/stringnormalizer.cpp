#include "StringNormalizer.h"
//cpp stuff
namespace backend {    
   
    StringNormalizer::StringNormalizer(const std::string& name) : Layer(name) { }
       
    vuh::Device* StringNormalizer::_get_device() {
        
        return device;
    }
    
    void StringNormalizer::init( int _case_change_action,  int _is_case_sensitive,  int _locale) {      
		 case_change_action = _case_change_action; 
 		 is_case_sensitive = _is_case_sensitive; 
 		 locale = _locale; 
  
    }
    
    void StringNormalizer::bind(std::string _stopwords, std::string _X_i, std::string _Y_o){
        stopwords = _stopwords; X_i = _X_i; Y_o = _Y_o;
		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.case_change_action = case_change_action;
  		binding.is_case_sensitive = is_case_sensitive;
  		binding.locale = locale;
 
		binding.stopwords = tensor_dict[stopwords]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/stringnormalizer.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[stopwords]->data(), *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
    }

}

