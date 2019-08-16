#include "StringNormalizer.h"

//cpp stuff
namespace backend {    
   
    StringNormalizer::StringNormalizer(std::string n, int case_change_action, int is_case_sensitive, int locale) : Layer(n) { }
       
    vuh::Device* StringNormalizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void StringNormalizer::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.case_change_action = case_change_action;
  		binding.is_case_sensitive = is_case_sensitive;
  		binding.locale = locale;
  		binding.stopwords = tensor_dict[stopwords]->shape();
 
    }
    
    void StringNormalizer::call(std::string stopwords, std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/stringnormalizer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[stopwords]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


