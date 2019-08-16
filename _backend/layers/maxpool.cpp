#include "MaxPool.h"

//cpp stuff
namespace backend {    
   
    MaxPool::MaxPool(std::string n) : Layer(n) { }
       
    vuh::Device* MaxPool::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void MaxPool::init( Shape_t _kernel_shape,  int _auto_pad,  int _ceil_mode,  Shape_t _dilations,  Shape_t _pads,  int _storage_order,  Shape_t _strides) {      
		 kernel_shape = _kernel_shape; 
 		 auto_pad = _auto_pad; 
 		 ceil_mode = _ceil_mode; 
 		 dilations = _dilations; 
 		 pads = _pads; 
 		 storage_order = _storage_order; 
 		 strides = _strides; 
  
    }
    
    void MaxPool::bind(std::string _X_input, std::string _Y_output, std::string _Indices_output_opt){
        X_input = _X_input; Y_output = _Y_output; Indices_output_opt = _Indices_output_opt;
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
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/maxpool.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data(), *tensor_dict[Indices_output_opt]->data());
    }
    
}

    //backend::nn;

//python stuff


