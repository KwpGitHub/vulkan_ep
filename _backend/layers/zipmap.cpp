#include "zipmap.h"
//cpp stuff
namespace layers {    
   
    ZipMap::ZipMap(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/zipmap.spv");       
        dev = backend::g_device;
    }
       
        
    void ZipMap::init( std::vector<int> _classlabels_int64s,  std::vector<std::string> _classlabels_strings) {      
		 m_classlabels_int64s = _classlabels_int64s; 
 		 m_classlabels_strings = _classlabels_strings; 
  

    }
    
    void ZipMap::bind(std::string _X_i, std::string _Z_o){    
        m_X_i = _X_i; m_Z_o = _Z_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Z_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void ZipMap::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Z_o]->data);       
    }

    void ZipMap::forward(){ 
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Z_o]->data);
        //program->run();
    }

}

