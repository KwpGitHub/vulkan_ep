#include "max.h"
//cpp stuff
namespace layers {    
   
    Max::Max(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/max.spv");       
        dev = backend::g_device;
    }
       
        
    void Max::init() {      
  

    }
    
    void Max::bind(std::string _max_o){    
        m_max_o = _max_o;        

		SHAPES.push_back(backend::tensor_dict[m_max_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Max::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        program->bind({2, 1}, *_SHAPES, *backend::tensor_dict[m_max_o]->data);
    }

    void Max::forward(){ 
        program->run();
    }

}

