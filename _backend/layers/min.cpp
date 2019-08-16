#include "Min.h"

//cpp stuff
namespace backend {    
   
    Min::Min(std::string n) : Layer(n) { }
       
    vuh::Device* Min::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Min::init() {      
    

		binding.min_output = tensor_dict[min_output]->shape();
 

    }
    
    void Min::call(std::string min_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/min.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[min_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


