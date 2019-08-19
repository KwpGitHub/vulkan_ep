#include "DictVectorizer.h"
//cpp stuff
namespace backend {    
   
    DictVectorizer::DictVectorizer() : Layer() { }
       
    vuh::Device* DictVectorizer::_get_device() {
        
        return device;
    }
    
    void DictVectorizer::init( Shape_t _int64_vocabulary) {      
		 int64_vocabulary = _int64_vocabulary; 
  
    }
    
    void DictVectorizer::bind(std::string _string_vocabulary, std::string _X_input, std::string _Y_output){
        string_vocabulary = _string_vocabulary; X_input = _X_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.int64_vocabulary = int64_vocabulary;
 
		binding.string_vocabulary = tensor_dict[string_vocabulary]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/dictvectorizer.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[string_vocabulary]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }



}



