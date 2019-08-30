#include "transpose.h"
//cpp stuff
namespace layers {    
   
    Transpose::Transpose(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/transpose.spv");       
        dev = backend::g_device;
    }
       
        
    void Transpose::init( std::vector<int> _perm) {      
		 m_perm = _perm; 
  

    }
    
    void Transpose::bind(std::string _data_i, std::string _transposed_o){    
        m_data_i = _data_i; m_transposed_o = _transposed_o;        
		SHAPES.push_back(backend::tensor_dict[m_data_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_transposed_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Transpose::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(SHAPES[0].w, SHAPES[0].h, SHAPES[0].d);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[m_data_i]->data, *backend::tensor_dict[m_transposed_o]->data);
    }

    void Transpose::forward(){ 
        program->run();
    }

}

