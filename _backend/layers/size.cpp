#include "size.h"
//cpp stuff
namespace layers {    
   
    Size::Size(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/size.spv");       
        dev = backend::g_device;
    }
       
        
    void Size::init() {      
  

    }
    
    void Size::bind(std::string _data_i, std::string _size_o){    
        m_data_i = _data_i; m_size_o = _size_o;        
		SHAPES.push_back(backend::tensor_dict[m_data_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_size_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Size::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        program->bind({2, 1}, *_SHAPES, *backend::tensor_dict[m_data_i]->data, *backend::tensor_dict[m_size_o]->data);
    }

    void Size::forward(){ 
        program->run();
    }

}

