#include "onehot.h"
//cpp stuff
namespace layers {    
   
    OneHot::OneHot(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/onehot.spv");       
        dev = backend::g_device;
    }
       
        
    void OneHot::init( int _axis) {      
		 m_axis = _axis; 
  

    }
    
    void OneHot::bind(std::string _indices_i, std::string _depth_i, std::string _values_i, std::string _output_o){    
        m_indices_i = _indices_i; m_depth_i = _depth_i; m_values_i = _values_i; m_output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[m_indices_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_depth_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_values_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void OneHot::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        program->bind({2, 1}, *_SHAPES, *backend::tensor_dict[m_indices_i]->data, *backend::tensor_dict[m_depth_i]->data, *backend::tensor_dict[m_values_i]->data, *backend::tensor_dict[m_output_o]->data);
    }

    void OneHot::forward(){ 
        program->run();
    }

}

