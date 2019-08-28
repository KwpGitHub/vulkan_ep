#include "dequantizelinear.h"
//cpp stuff
namespace layers {    
   
    DequantizeLinear::DequantizeLinear(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/dequantizelinear.spv");       
        dev = backend::device;
    }
       
        
    void DequantizeLinear::init() {      
  

    }
    
    void DequantizeLinear::bind(std::string _x_i, std::string _x_scale_i, std::string _x_zero_point_i, std::string _y_o){    
        x_i = _x_i; x_scale_i = _x_scale_i; x_zero_point_i = _x_zero_point_i; y_o = _y_o;        
		SHAPES.push_back(backend::tensor_dict[x_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x_zero_point_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void DequantizeLinear::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[x_i]->data, *backend::tensor_dict[x_scale_i]->data, *backend::tensor_dict[x_zero_point_i]->data, *backend::tensor_dict[y_o]->data);
    }

    void DequantizeLinear::forward(){ 
        program->run();
    }

}

