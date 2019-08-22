#include "conv.h"
//cpp stuff
namespace layers {    
   
    Conv::Conv(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\conv.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* Conv::_get_device() {
        
        return backend::device;
    }
    
    void Conv::init( std::string _auto_pad,  std::vector<int> _dilations,  int _group,  std::vector<int> _kernel_shape,  std::vector<int> _pads,  std::vector<int> _strides) {      
		 auto_pad = _auto_pad; 
 		 dilations = _dilations; 
 		 group = _group; 
 		 kernel_shape = _kernel_shape; 
 		 pads = _pads; 
 		 strides = _strides; 
  
    }
    
    void Conv::bind(std::string _X_i, std::string _W_i, std::string _B_i, std::string _Y_o){
        X_i = _X_i; W_i = _W_i; B_i = _B_i; Y_o = _Y_o;

		//binding.X_i = tensor_dict[X_i]->shape();
  		//binding.W_i = tensor_dict[W_i]->shape();
  		//binding.B_i = tensor_dict[B_i]->shape();
 
		//binding.Y_o = tensor_dict[Y_o]->shape();
 
		//binding.auto_pad = auto_pad;
  		//binding.dilations = dilations;
  		//binding.group = group;
  		//binding.kernel_shape = kernel_shape;
  		//binding.pads = pads;
  		//binding.strides = strides;
         
    }

    void Conv::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[W_i]->data(), *tensor_dict[B_i]->data(), *tensor_dict[Y_o]->data());
    }

}

