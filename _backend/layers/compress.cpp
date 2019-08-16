#include "Compress.h"

//cpp stuff
namespace backend {    
   
    Compress::Compress(std::string n, int axis) : Layer(n) { }
       
    vuh::Device* Compress::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Compress::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
  		binding.condition_input = tensor_dict[condition_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.axis = axis;
 
    }
    
    void Compress::call(std::string input_input, std::string condition_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/compress.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[condition_input]->data(), *tensor_dict[output_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


