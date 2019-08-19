#include "QLinearConv.h"
//cpp stuff
namespace backend {    
   
    QLinearConv::QLinearConv() : Layer() { }
       
    vuh::Device* QLinearConv::_get_device() {
        
        return device;
    }
    
    void QLinearConv::init( int _auto_pad,  Shape_t _dilations,  int _group,  Shape_t _kernel_shape,  Shape_t _pads,  Shape_t _strides) {      
		 auto_pad = _auto_pad; 
 		 dilations = _dilations; 
 		 group = _group; 
 		 kernel_shape = _kernel_shape; 
 		 pads = _pads; 
 		 strides = _strides; 
  
    }
    
    void QLinearConv::bind(std::string _x_input, std::string _x_scale_input, std::string _x_zero_point_input, std::string _w_input, std::string _w_scale_input, std::string _w_zero_point_input, std::string _y_scale_input, std::string _y_zero_point_input, std::string _B_input_opt, std::string _y_output){
        x_input = _x_input; x_scale_input = _x_scale_input; x_zero_point_input = _x_zero_point_input; w_input = _w_input; w_scale_input = _w_scale_input; w_zero_point_input = _w_zero_point_input; y_scale_input = _y_scale_input; y_zero_point_input = _y_zero_point_input; B_input_opt = _B_input_opt; y_output = _y_output;
		binding.x_input = tensor_dict[x_input]->shape();
  		binding.x_scale_input = tensor_dict[x_scale_input]->shape();
  		binding.x_zero_point_input = tensor_dict[x_zero_point_input]->shape();
  		binding.w_input = tensor_dict[w_input]->shape();
  		binding.w_scale_input = tensor_dict[w_scale_input]->shape();
  		binding.w_zero_point_input = tensor_dict[w_zero_point_input]->shape();
  		binding.y_scale_input = tensor_dict[y_scale_input]->shape();
  		binding.y_zero_point_input = tensor_dict[y_zero_point_input]->shape();
  		binding.B_input_opt = tensor_dict[B_input_opt]->shape();
 
		binding.y_output = tensor_dict[y_output]->shape();
 
		binding.auto_pad = auto_pad;
  		binding.dilations = dilations;
  		binding.group = group;
  		binding.kernel_shape = kernel_shape;
  		binding.pads = pads;
  		binding.strides = strides;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/qlinearconv.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[x_input]->data(), *tensor_dict[x_scale_input]->data(), *tensor_dict[x_zero_point_input]->data(), *tensor_dict[w_input]->data(), *tensor_dict[w_scale_input]->data(), *tensor_dict[w_zero_point_input]->data(), *tensor_dict[y_scale_input]->data(), *tensor_dict[y_zero_point_input]->data(), *tensor_dict[B_input_opt]->data(), *tensor_dict[y_output]->data());
    }



}



