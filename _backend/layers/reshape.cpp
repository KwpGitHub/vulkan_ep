#include "reshape.h"
//cpp stuff
namespace layers {    
   
    Reshape::Reshape(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/reshape.spv");       
        dev = backend::g_device;
    }
       
        
    void Reshape::init() {      
  

    }
    
    void Reshape::bind(std::string _data_i, std::string _shape_i, std::string _reshaped_o){    
        m_data_i = _data_i; m_shape_i = _shape_i; m_reshaped_o = _reshaped_o;        
		SHAPES.push_back(backend::tensor_dict[m_data_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_shape_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_reshaped_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Reshape::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_data_i]->data, *backend::tensor_dict[m_shape_i]->data, *backend::tensor_dict[m_reshaped_o]->data);
    }

    void Reshape::forward(){ 
        program->run();
    }

}

