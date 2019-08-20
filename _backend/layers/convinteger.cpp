#include "ConvInteger.h"
//cpp stuff
namespace backend {    
   
    ConvInteger::ConvInteger(const std::string& name) : Layer(name) { }
       
    vuh::Device* ConvInteger::_get_device() {
        
        return device;
    }
    
    void ConvInteger::init( int _auto_pad,  Shape_t _dilations,  int _group,  Shape_t _kernel_shape,  Shape_t _pads,  Shape_t _strides) {      
		 auto_pad = _auto_pad; 
 		 dilations = _dilations; 
 		 group = _group; 
 		 kernel_shape = _kernel_shape; 
 		 pads = _pads; 
 		 strides = _strides; 
  
    }
    
    void ConvInteger::bind(std::string _x_i, std::string _w_i, std::string _x_zero_point_i, std::string _w_zero_point_i, std::string _y_o){
        x_i = _x_i; w_i = _w_i; x_zero_point_i = _x_zero_point_i; w_zero_point_i = _w_zero_point_i; y_o = _y_o;
		binding.x_i = tensor_dict[x_i]->shape();
  		binding.w_i = tensor_dict[w_i]->shape();
  		binding.x_zero_point_i = tensor_dict[x_zero_point_i]->shape();
  		binding.w_zero_point_i = tensor_dict[w_zero_point_i]->shape();
 
		binding.y_o = tensor_dict[y_o]->shape();
 
		binding.auto_pad = auto_pad;
  		binding.dilations = dilations;
  		binding.group = group;
  		binding.kernel_shape = kernel_shape;
  		binding.pads = pads;
  		binding.strides = strides;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/convinteger.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[x_i]->data(), *tensor_dict[w_i]->data(), *tensor_dict[x_zero_point_i]->data(), *tensor_dict[w_zero_point_i]->data(), *tensor_dict[y_o]->data());
    }

}

