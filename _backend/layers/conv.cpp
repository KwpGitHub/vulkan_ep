#include "Conv.h"

//cpp stuff
namespace backend {    
   
    Conv::Conv(std::string n, int auto_pad, Shape_t dilations, int group, Shape_t kernel_shape, Shape_t pads, Shape_t strides) : Layer(n) { }
       
    vuh::Device* Conv::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Conv::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.W_input = tensor_dict[W_input]->shape();
  		binding.B_input_opt = tensor_dict[B_input_opt]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.auto_pad = auto_pad;
  		binding.dilations = dilations;
  		binding.group = group;
  		binding.kernel_shape = kernel_shape;
  		binding.pads = pads;
  		binding.strides = strides;
 
    }
    
    void Conv::call(std::string X_input, std::string W_input, std::string B_input_opt, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/conv.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[W_input]->data(), *tensor_dict[B_input_opt]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


