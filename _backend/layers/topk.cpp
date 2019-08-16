#include "TopK.h"

//cpp stuff
namespace backend {    
   
    TopK::TopK(std::string n, int axis) : Layer(n) { }
       
    vuh::Device* TopK::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void TopK::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.K_input = tensor_dict[K_input]->shape();
 
		binding.Values_output = tensor_dict[Values_output]->shape();
  		binding.Indices_output = tensor_dict[Indices_output]->shape();
 
		binding.axis = axis;
 
    }
    
    void TopK::call(std::string X_input, std::string K_input, std::string Values_output, std::string Indices_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/topk.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[K_input]->data(), *tensor_dict[Values_output]->data(), *tensor_dict[Indices_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


