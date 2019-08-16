#include "If.h"

//cpp stuff
namespace backend {    
   
    If::If(std::string n, int else_branch, int then_branch) : Layer(n) { }
       
    vuh::Device* If::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void If::init() {      
    
		binding.cond_input = tensor_dict[cond_input]->shape();
 

		binding.else_branch = else_branch;
  		binding.then_branch = then_branch;
 
    }
    
    void If::call(std::string cond_input){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/if.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[cond_input]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


