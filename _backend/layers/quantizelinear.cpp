#include "quantizelinear.h"
//cpp stuff
namespace layers {    
   
    QuantizeLinear::QuantizeLinear(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/quantizelinear.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* QuantizeLinear::_get_device() {        
        return backend::device;
    }
    
    void QuantizeLinear::init() {      
  
    }
    
    void QuantizeLinear::bind(std::string _x_i, std::string _y_scale_i, std::string _y_zero_point_i, std::string _y_o){
        x_i = _x_i; y_scale_i = _y_scale_i; y_zero_point_i = _y_zero_point_i; y_o = _y_o;

		binding.x_i = backend::tensor_dict[x_i]->shape();
  		binding.y_scale_i = backend::tensor_dict[y_scale_i]->shape();
  		binding.y_zero_point_i = backend::tensor_dict[y_zero_point_i]->shape();
 
		binding.y_o = backend::tensor_dict[y_o]->shape();
 
        
    }

    void QuantizeLinear::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[x_i]->data(), *backend::tensor_dict[y_scale_i]->data(), *backend::tensor_dict[y_zero_point_i]->data(), *backend::tensor_dict[y_o]->data());
    }

    void QuantizeLinear::forward(){ 
        //program->run();
    }

}

