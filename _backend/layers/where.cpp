#include "Where.h"
//cpp stuff
namespace backend {    
   
    Where::Where(const std::string& name) : Layer(name) { }
       
    vuh::Device* Where::_get_device() {
        
        return device;
    }
    
    void Where::init() {      
  
    }
    
    void Where::bind(std::string _condition_i, std::string _X_i, std::string _Y_i, std::string _output_o){
        condition_i = _condition_i; X_i = _X_i; Y_i = _Y_i; output_o = _output_o;
		binding.condition_i = tensor_dict[condition_i]->shape();
  		binding.X_i = tensor_dict[X_i]->shape();
  		binding.Y_i = tensor_dict[Y_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/where.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[condition_i]->data(), *tensor_dict[X_i]->data(), *tensor_dict[Y_i]->data(), *tensor_dict[output_o]->data());
    }

}

