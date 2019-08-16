#include "AveragePool.h"

//cpp stuff
namespace backend {    
   
    AveragePool::AveragePool(std::string n, Shape_t kernel_shape, int auto_pad, int ceil_mode, int count_include_pad, Shape_t pads, Shape_t strides) : Layer(n) { }
       
    vuh::Device* AveragePool::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void AveragePool::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.kernel_shape = kernel_shape;
  		binding.auto_pad = auto_pad;
  		binding.ceil_mode = ceil_mode;
  		binding.count_include_pad = count_include_pad;
  		binding.pads = pads;
  		binding.strides = strides;
 
    }
    
    void AveragePool::call(std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/averagepool.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


