#include "slice.h"
//cpp stuff
namespace layers {    
   
    Slice::Slice(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/slice.spv");       
        dev = backend::g_device;
    }
       
        
    void Slice::init() {      
  

    }
    
    void Slice::bind(std::string _data_i, std::string _starts_i, std::string _ends_i, std::string _axes_i, std::string _steps_i, std::string _output_o){    
        m_data_i = _data_i; m_starts_i = _starts_i; m_ends_i = _ends_i; m_axes_i = _axes_i; m_steps_i = _steps_i; m_output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[m_data_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_starts_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_ends_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_axes_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_steps_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Slice::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_data_i]->data, *backend::tensor_dict[m_starts_i]->data, *backend::tensor_dict[m_ends_i]->data, *backend::tensor_dict[m_axes_i]->data, *backend::tensor_dict[m_steps_i]->data, *backend::tensor_dict[m_output_o]->data);
    }

    void Slice::forward(){ 
        program->run();
    }

}

