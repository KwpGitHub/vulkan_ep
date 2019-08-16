#include "StringNormalizer.h"

//cpp stuff
namespace backend {    
   
    StringNormalizer::StringNormalizer(std::string n) : Layer(n) { }
       
    vuh::Device* StringNormalizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void StringNormalizer::init( int _case_change_action,  int _is_case_sensitive,  int _locale) {      
		 case_change_action = _case_change_action; 
 		 is_case_sensitive = _is_case_sensitive; 
 		 locale = _locale; 
  
    }
    
    void StringNormalizer::bind(std::string _stopwords, std::string _X_input, std::string _Y_output){
        stopwords = _stopwords; X_input = _X_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.case_change_action = case_change_action;
  		binding.is_case_sensitive = is_case_sensitive;
  		binding.locale = locale;
 
		binding.stopwords = tensor_dict[stopwords]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/stringnormalizer.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[stopwords]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    //backend::nn;

//python stuff


