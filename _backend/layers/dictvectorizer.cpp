#include "DictVectorizer.h"
//cpp stuff
namespace backend {    
   
    DictVectorizer::DictVectorizer(const std::string& name) : Layer(name) { }
       
    vuh::Device* DictVectorizer::_get_device() {
        
        return device;
    }
    
    void DictVectorizer::init( Shape_t _int64_vocabulary) {      
		 int64_vocabulary = _int64_vocabulary; 
  
    }
    
    void DictVectorizer::bind(std::string _string_vocabulary, std::string _X_i, std::string _Y_o){
        string_vocabulary = _string_vocabulary; X_i = _X_i; Y_o = _Y_o;
		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.int64_vocabulary = int64_vocabulary;
 
		binding.string_vocabulary = tensor_dict[string_vocabulary]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/dictvectorizer.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[string_vocabulary]->data(), *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
    }

}

