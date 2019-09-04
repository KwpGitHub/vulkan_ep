#include "categorymapper.h"
//cpp stuff
namespace layers {    
   
    CategoryMapper::CategoryMapper(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/categorymapper.spv");       
        dev = backend::g_device;
    }
       
        
    void CategoryMapper::init( std::vector<int> _cats_int64s,  std::vector<std::string> _cats_strings,  int _default_int64,  std::string _default_string) {      
		 m_cats_int64s = _cats_int64s; 
 		 m_cats_strings = _cats_strings; 
 		 m_default_int64 = _default_int64; 
 		 m_default_string = _default_string; 
  

    }
    
    void CategoryMapper::bind(std::string _X_i, std::string _Y_o){    
        m_X_i = _X_i; m_Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void CategoryMapper::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
       
    }

    void CategoryMapper::forward(){ 
        program->operator()({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data);
        //program->run();
    }

}

