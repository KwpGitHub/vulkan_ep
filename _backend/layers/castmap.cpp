#include "castmap.h"
//cpp stuff
namespace layers {    
   
    CastMap::CastMap(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/castmap.spv");       
        dev = backend::g_device;
    }
       
        
    void CastMap::init( std::string _cast_to,  std::string _map_form,  int _max_map) {      
		 m_cast_to = _cast_to; 
 		 m_map_form = _map_form; 
 		 m_max_map = _max_map; 
  

    }
    
    void CastMap::bind(std::string _X_i, std::string _Y_o){    
        m_X_i = _X_i; m_Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void CastMap::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data);       
    }

    void CastMap::forward(){ 
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data);
        //program->run();
    }

}

