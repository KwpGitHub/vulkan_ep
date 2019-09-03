#include "min.h"
//cpp stuff
namespace layers {    
   
    Min::Min(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/min.spv");       
        dev = backend::g_device;
    }
       
        
    void Min::init() {      
  

    }
    
    void Min::bind(std::string _min_o){    
        m_min_o = _min_o;        

		SHAPES.push_back(backend::tensor_dict[m_min_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Min::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        program->bind({2, 1}, *_SHAPES, *backend::tensor_dict[m_min_o]->data);
    }

    void Min::forward(){ 
        program->run();
    }

}

