#include "Loop.h"

//cpp stuff
namespace backend {    
   
    Loop::Loop(std::string n, int body) : Layer(n) { }
       
    vuh::Device* Loop::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Loop::init() {      
    
		binding.M_input_opt = tensor_dict[M_input_opt]->shape();
  		binding.cond_input_opt = tensor_dict[cond_input_opt]->shape();
 

		binding.body = body;
 
    }
    
    void Loop::call(std::string M_input_opt, std::string cond_input_opt){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/loop.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[M_input_opt]->data(), *tensor_dict[cond_input_opt]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


