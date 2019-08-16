#include "MaxPool.h"

//cpp stuff
namespace backend {    
   
    MaxPool::MaxPool(std::string n, Shape_t kernel_shape, int auto_pad, int ceil_mode, Shape_t dilations, Shape_t pads, int storage_order, Shape_t strides) : Layer(n) { }
       
    vuh::Device* MaxPool::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void MaxPool::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
  		binding.Indices_output_opt = tensor_dict[Indices_output_opt]->shape();
 
		binding.kernel_shape = kernel_shape;
  		binding.auto_pad = auto_pad;
  		binding.ceil_mode = ceil_mode;
  		binding.dilations = dilations;
  		binding.pads = pads;
  		binding.storage_order = storage_order;
  		binding.strides = strides;
 
    }
    
    void MaxPool::call(std::string X_input, std::string Y_output, std::string Indices_output_opt){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/maxpool.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data(), *tensor_dict[Indices_output_opt]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


