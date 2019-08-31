#include "where.h"
//cpp stuff
namespace layers {    
   
    Where::Where(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/where.spv");       
        dev = backend::g_device;
    }
       
        
    void Where::init() {      
  

    }
    
    void Where::bind(std::string _condition_i, std::string _X_i, std::string _Y_i, std::string _output_o){    
        m_condition_i = _condition_i; m_X_i = _X_i; m_Y_i = _Y_i; m_output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[m_condition_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_Y_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Where::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_condition_i]->data, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_i]->data, *backend::tensor_dict[m_output_o]->data);
    }

    void Where::forward(){ 
        program->run();
    }

}

