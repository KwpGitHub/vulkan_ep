#include "globalaveragepool.h"
//cpp stuff
namespace layers {    
   
    GlobalAveragePool::GlobalAveragePool(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/globalaveragepool.spv");       
        dev = backend::g_device;
    }
       
        
    void GlobalAveragePool::init() {      
  

    }
    
    void GlobalAveragePool::bind(std::string _X_i, std::string _Y_o){    
        m_X_i = _X_i; m_Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void GlobalAveragePool::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        program->bind({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data);
    }

    void GlobalAveragePool::forward(){ 
        program->run();
    }

}

