#include "averagepool.h"
//cpp stuff
namespace layers {    
   
    AveragePool::AveragePool(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/averagepool.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* AveragePool::_get_device() {        
        return backend::device;
    }
    
    void AveragePool::init( std::vector<int> _kernel_shape,  std::string _auto_pad,  int _ceil_mode,  int _count_include_pad,  std::vector<int> _pads,  std::vector<int> _strides) {      
		 kernel_shape = _kernel_shape; 
 		 auto_pad = _auto_pad; 
 		 ceil_mode = _ceil_mode; 
 		 count_include_pad = _count_include_pad; 
 		 pads = _pads; 
 		 strides = _strides; 
  
    }
    
    void AveragePool::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.kernel_shape = kernel_shape;
  		//binding.auto_pad = auto_pad;
  		//binding.ceil_mode = ceil_mode;
  		//binding.count_include_pad = count_include_pad;
  		//binding.pads = pads;
  		//binding.strides = strides;
         
    }

    void AveragePool::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void AveragePool::forward(){ 
        program->run();
    }

}

