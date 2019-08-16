#include "QLinearConv.h"

//cpp stuff
namespace backend {    
   
    QLinearConv::QLinearConv(std::string n, int auto_pad, Shape_t dilations, int group, Shape_t kernel_shape, Shape_t pads, Shape_t strides) : Layer(n) { }
       
    vuh::Device* QLinearConv::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void QLinearConv::init() {      
    
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
 
    }
    
    void QLinearConv::call(std::string x_input, std::string x_scale_input, std::string x_zero_point_input, std::string w_input, std::string w_scale_input, std::string w_zero_point_input, std::string y_scale_input, std::string y_zero_point_input, std::string B_input_opt, std::string y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/qlinearconv.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[x_input]->data(), *tensor_dict[x_scale_input]->data(), *tensor_dict[x_zero_point_input]->data(), *tensor_dict[w_input]->data(), *tensor_dict[w_scale_input]->data(), *tensor_dict[w_zero_point_input]->data(), *tensor_dict[y_scale_input]->data(), *tensor_dict[y_zero_point_input]->data(), *tensor_dict[B_input_opt]->data(), *tensor_dict[y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


