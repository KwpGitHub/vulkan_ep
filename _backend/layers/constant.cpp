#include "Constant.h"

//cpp stuff
namespace backend {    
   
    Constant::Constant(std::string n) : Layer(n) { }
       
    vuh::Device* Constant::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Constant::init() {      
  
    }
    
    void Constant::bind(std::string _value, std::string _output_output){
        value = _value; output_output = _output_output;

		binding.output_output = tensor_dict[output_output]->shape();
 

		binding.value = tensor_dict[value]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/constant.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[value]->data(), *tensor_dict[output_output]->data());
    }
    
}

    //backend::nn;

//python stuff


