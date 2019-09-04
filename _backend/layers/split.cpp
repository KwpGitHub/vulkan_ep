#include "split.h"
//cpp stuff
namespace layers {    
   
    Split::Split(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/split.spv");       
        dev = backend::g_device;
    }
       
        
    void Split::init( int _axis,  std::vector<int> _split) {      
		 m_axis = _axis; 
 		 m_split = _split; 
  

    }
    
    void Split::bind(std::string _input_i){    
        m_input_i = _input_i;        
		SHAPES.push_back(backend::tensor_dict[m_input_i]->shape());
 

        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Split::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
       
    }

    void Split::forward(){ 
        program->operator()({2, 1}, *_SHAPES, *backend::tensor_dict[m_input_i]->data);
        //program->run();
    }

}

