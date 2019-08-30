#include "quantizelinear.h"
//cpp stuff
namespace layers {    
   
    QuantizeLinear::QuantizeLinear(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/quantizelinear.spv");       
        dev = backend::g_device;
    }
       
        
    void QuantizeLinear::init() {      
  

    }
    
    void QuantizeLinear::bind(std::string _x_i, std::string _y_scale_i, std::string _y_zero_point_i, std::string _y_o){    
        m_x_i = _x_i; m_y_scale_i = _y_scale_i; m_y_zero_point_i = _y_zero_point_i; m_y_o = _y_o;        
		SHAPES.push_back(backend::tensor_dict[m_x_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_y_zero_point_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void QuantizeLinear::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(SHAPES[0].w, SHAPES[0].h, SHAPES[0].d);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[m_x_i]->data, *backend::tensor_dict[m_y_scale_i]->data, *backend::tensor_dict[m_y_zero_point_i]->data, *backend::tensor_dict[m_y_o]->data);
    }

    void QuantizeLinear::forward(){ 
        program->run();
    }

}

