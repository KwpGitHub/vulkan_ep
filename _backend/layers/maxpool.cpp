#include "MaxPool.h"
//cpp stuff
namespace backend {    
   
    MaxPool::MaxPool(const std::string& name) : Layer(name) { }
       
    vuh::Device* MaxPool::_get_device() {
        
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
    
    void MaxPool::bind(std::string _X_i, std::string _Y_o, std::string _Indices_o){
        X_i = _X_i; Y_o = _Y_o; Indices_o = _Indices_o;
		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
  		binding.Indices_o = tensor_dict[Indices_o]->shape();
 
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
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data(), *tensor_dict[Indices_o]->data());
    }

}

