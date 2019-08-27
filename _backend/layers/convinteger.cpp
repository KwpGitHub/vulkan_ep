#include "convinteger.h"
//cpp stuff
namespace layers {    
   
    ConvInteger::ConvInteger(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/convinteger.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* ConvInteger::_get_device() {        
        return backend::device;
    }
    
    void ConvInteger::init( std::string _auto_pad,  std::vector<int> _dilations,  int _group,  std::vector<int> _kernel_shape,  std::vector<int> _pads,  std::vector<int> _strides) {      
		 auto_pad = _auto_pad; 
 		 dilations = _dilations; 
 		 group = _group; 
 		 kernel_shape = _kernel_shape; 
 		 pads = _pads; 
 		 strides = _strides; 
  
    }
    
    void ConvInteger::bind(std::string _x_i, std::string _w_i, std::string _x_zero_point_i, std::string _w_zero_point_i, std::string _y_o){
        x_i = _x_i; w_i = _w_i; x_zero_point_i = _x_zero_point_i; w_zero_point_i = _w_zero_point_i; y_o = _y_o;

		binding.x_i = backend::tensor_dict[x_i]->shape();
  		binding.w_i = backend::tensor_dict[w_i]->shape();
  		binding.x_zero_point_i = backend::tensor_dict[x_zero_point_i]->shape();
  		binding.w_zero_point_i = backend::tensor_dict[w_zero_point_i]->shape();
 
		binding.y_o = backend::tensor_dict[y_o]->shape();
 
		//binding.auto_pad = auto_pad;
  		//binding.dilations = dilations;
  		//binding.group = group;
  		//binding.kernel_shape = kernel_shape;
  		//binding.pads = pads;
  		//binding.strides = strides;
         
    }

    void ConvInteger::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[x_i]->data(), *backend::tensor_dict[w_i]->data(), *backend::tensor_dict[x_zero_point_i]->data(), *backend::tensor_dict[w_zero_point_i]->data(), *backend::tensor_dict[y_o]->data());
    }

    void ConvInteger::forward(){ 
        program->run();
    }

}

