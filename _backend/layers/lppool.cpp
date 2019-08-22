#include "lppool.h"
//cpp stuff
namespace layers {    
   
    LpPool::LpPool(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\lppool.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* LpPool::_get_device() {
        
        return backend::device;
    }
    
    void LpPool::init( std::vector<int> _kernel_shape,  std::string _auto_pad,  int _p,  std::vector<int> _pads,  std::vector<int> _strides) {      
		 kernel_shape = _kernel_shape; 
 		 auto_pad = _auto_pad; 
 		 p = _p; 
 		 pads = _pads; 
 		 strides = _strides; 
  
    }
    
    void LpPool::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		//binding.X_i = tensor_dict[X_i]->shape();
 
		//binding.Y_o = tensor_dict[Y_o]->shape();
 
		//binding.kernel_shape = kernel_shape;
  		//binding.auto_pad = auto_pad;
  		//binding.p = p;
  		//binding.pads = pads;
  		//binding.strides = strides;
         
    }

    void LpPool::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
    }

}

