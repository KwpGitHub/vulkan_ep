#include "gather.h"
//cpp stuff
namespace layers {    
   
    Gather::Gather(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/gather.spv");       
        dev = backend::g_device;
    }
       
        
    void Gather::init( int _axis) {      
		 m_axis = _axis; 
  

    }
    
    void Gather::bind(std::string _data_i, std::string _indices_i, std::string _output_o){    
        m_data_i = _data_i; m_indices_i = _indices_i; m_output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[m_data_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_indices_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Gather::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
       
    }

    void Gather::forward(){ 
        program->operator()({2, 1}, *_SHAPES, *backend::tensor_dict[m_data_i]->data, *backend::tensor_dict[m_indices_i]->data, *backend::tensor_dict[m_output_o]->data);
        //program->run();
    }

}

