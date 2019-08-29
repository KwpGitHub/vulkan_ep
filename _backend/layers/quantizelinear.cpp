#include "quantizelinear.h"
//cpp stuff
namespace layers {    
   
    QuantizeLinear::QuantizeLinear(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/quantizelinear.spv");       
        dev = backend::device;
    }
       
        
    void QuantizeLinear::init() {      
  

    }
    
    void QuantizeLinear::bind(std::string _x_i, std::string _y_scale_i, std::string _y_zero_point_i, std::string _y_o){    
        x_i = _x_i; y_scale_i = _y_scale_i; y_zero_point_i = _y_zero_point_i; y_o = _y_o;        
		SHAPES.push_back(backend::tensor_dict[x_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[y_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[y_zero_point_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void QuantizeLinear::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[x_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[x_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[x_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[x_i]->data, *backend::tensor_dict[y_scale_i]->data, *backend::tensor_dict[y_zero_point_i]->data, *backend::tensor_dict[y_o]->data);
    }

    void QuantizeLinear::forward(){ 
        program->run();
    }

}

