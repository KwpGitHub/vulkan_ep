#include "constant.h"
//cpp stuff
namespace layers {    
   
    Constant::Constant(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/constant.spv");       
        dev = backend::g_device;
    }
       
        
    void Constant::init( std::vector<float> _value) {      
		 m_value = _value; 
  

    }
    
    void Constant::bind(std::string _output_o){    
        m_output_o = _output_o;        

		SHAPES.push_back(backend::tensor_dict[m_output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Constant::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_output_o]->data);       
    }

    void Constant::forward(){ 
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_output_o]->data);
        //program->run();
    }

}

