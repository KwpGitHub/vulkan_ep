#include "add.h"
//cpp stuff
namespace layers {    
   
    Add::Add(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/add.spv");       
        dev = backend::g_device;
    }
       
        
    void Add::init() {      
  

    }
    
    void Add::bind(std::string _A_i, std::string _B_i, std::string _C_o){    
        m_A_i = _A_i; m_B_i = _B_i; m_C_o = _C_o;        
		SHAPES.push_back(backend::tensor_dict[m_A_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_B_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_C_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Add::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
       
    }

    void Add::forward(){ 
        program->operator()({2, 1}, *_SHAPES, *backend::tensor_dict[m_A_i]->data, *backend::tensor_dict[m_B_i]->data, *backend::tensor_dict[m_C_o]->data);
        //program->run();
    }

}

