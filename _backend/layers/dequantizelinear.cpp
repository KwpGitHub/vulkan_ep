#include "dequantizelinear.h"
//cpp stuff
namespace layers {    
   
    DequantizeLinear::DequantizeLinear(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\dequantizelinear.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* DequantizeLinear::_get_device() {
        
        return backend::device;
    }
    
    void DequantizeLinear::init() {      
  
    }
    
    void DequantizeLinear::bind(std::string _x_i, std::string _x_scale_i, std::string _x_zero_point_i, std::string _y_o){
        x_i = _x_i; x_scale_i = _x_scale_i; x_zero_point_i = _x_zero_point_i; y_o = _y_o;

		//binding.x_i = tensor_dict[x_i]->shape();
  		//binding.x_scale_i = tensor_dict[x_scale_i]->shape();
  		//binding.x_zero_point_i = tensor_dict[x_zero_point_i]->shape();
 
		//binding.y_o = tensor_dict[y_o]->shape();
 
        
    }

    void DequantizeLinear::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[x_i]->data(), *tensor_dict[x_scale_i]->data(), *tensor_dict[x_zero_point_i]->data(), *tensor_dict[y_o]->data());
    }

}

