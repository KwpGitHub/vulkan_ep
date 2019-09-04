#include "pad.h"
//cpp stuff
namespace layers {    
   
    Pad::Pad(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/pad.spv");       
        dev = backend::g_device;
    }
       
        
    void Pad::init( std::vector<int> _pads,  std::string _mode,  float _value) {      
		 m_pads = _pads; 
 		 m_mode = _mode; 
 		 m_value = _value; 
  

    }
    
    void Pad::bind(std::string _data_i, std::string _output_o){    
        m_data_i = _data_i; m_output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[m_data_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Pad::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
       
    }

    void Pad::forward(){ 
        program->operator()({2, 1}, *_SHAPES, *backend::tensor_dict[m_data_i]->data, *backend::tensor_dict[m_output_o]->data);
        //program->run();
    }

}

