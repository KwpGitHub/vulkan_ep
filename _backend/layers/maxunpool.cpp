#include "maxunpool.h"
//cpp stuff
namespace layers {    
   
    MaxUnpool::MaxUnpool(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\maxunpool.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* MaxUnpool::_get_device() {
        
        return backend::device;
    }
    
    void MaxUnpool::init( std::vector<int> _kernel_shape,  std::vector<int> _pads,  std::vector<int> _strides) {      
		 kernel_shape = _kernel_shape; 
 		 pads = _pads; 
 		 strides = _strides; 
  
    }
    
    void MaxUnpool::bind(std::string _X_i, std::string _I_i, std::string _output_shape_i, std::string _output_o){
        X_i = _X_i; I_i = _I_i; output_shape_i = _output_shape_i; output_o = _output_o;

		//binding.X_i = tensor_dict[X_i]->shape();
  		//binding.I_i = tensor_dict[I_i]->shape();
  		//binding.output_shape_i = tensor_dict[output_shape_i]->shape();
 
		//binding.output_o = tensor_dict[output_o]->shape();
 
		//binding.kernel_shape = kernel_shape;
  		//binding.pads = pads;
  		//binding.strides = strides;
         
    }

    void MaxUnpool::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[I_i]->data(), *tensor_dict[output_shape_i]->data(), *tensor_dict[output_o]->data());
    }

}

