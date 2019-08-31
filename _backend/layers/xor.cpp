#include "xor.h"
//cpp stuff
namespace layers {    
   
    Xor::Xor(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/xor.spv");       
        dev = backend::g_device;
    }
       
        
    void Xor::init() {      
  

    }
    
    void Xor::bind(std::string _A_i, std::string _B_i, std::string _C_o){    
        m_A_i = _A_i; m_B_i = _B_i; m_C_o = _C_o;        
		SHAPES.push_back(backend::tensor_dict[m_A_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_B_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_C_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Xor::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_A_i]->data, *backend::tensor_dict[m_B_i]->data, *backend::tensor_dict[m_C_o]->data);
    }

    void Xor::forward(){ 
        program->run();
    }

}

