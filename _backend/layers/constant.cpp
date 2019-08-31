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
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_output_o]->data);
    }

    void Constant::forward(){ 
        program->run();
    }

}

