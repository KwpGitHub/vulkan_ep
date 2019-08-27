#include "maxpool.h"
//cpp stuff
namespace layers {    
   
    MaxPool::MaxPool(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/maxpool.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* MaxPool::_get_device() {        
        return backend::device;
    }
    
    void MaxPool::init( std::vector<int> _kernel_shape,  std::string _auto_pad,  int _ceil_mode,  std::vector<int> _dilations,  std::vector<int> _pads,  int _storage_order,  std::vector<int> _strides) {      
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

		binding.X_i = backend::tensor_dict[X_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
  		binding.Indices_o = backend::tensor_dict[Indices_o]->shape();
 
		//binding.kernel_shape = kernel_shape;
  		//binding.auto_pad = auto_pad;
  		//binding.ceil_mode = ceil_mode;
  		//binding.dilations = dilations;
  		//binding.pads = pads;
  		//binding.storage_order = storage_order;
  		//binding.strides = strides;
         
    }

    void MaxPool::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_o]->data(), *backend::tensor_dict[Indices_o]->data());
    }

    void MaxPool::forward(){ 
        //program->run();
    }

}

