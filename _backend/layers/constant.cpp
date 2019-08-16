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
    

		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.value = tensor_dict[value]->shape();
 
    }
    
    void Constant::call(std::string value, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/constant.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[value]->data(), *tensor_dict[output_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


