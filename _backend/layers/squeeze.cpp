#include "squeeze.h"
//cpp stuff
namespace layers {    
   
    Squeeze::Squeeze(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/squeeze.spv");       
        dev = backend::g_device;
    }
       
        
    void Squeeze::init( std::vector<int> _axes) {      
		 m_axes = _axes; 
  

    }
    
    void Squeeze::bind(std::string _data_i, std::string _squeezed_o){    
        m_data_i = _data_i; m_squeezed_o = _squeezed_o;        
		SHAPES.push_back(backend::tensor_dict[m_data_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_squeezed_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Squeeze::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
       
    }

    void Squeeze::forward(){ 
        program->operator()({2, 1}, *_SHAPES, *backend::tensor_dict[m_data_i]->data, *backend::tensor_dict[m_squeezed_o]->data);
        //program->run();
    }

}

