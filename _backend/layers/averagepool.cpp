#include "averagepool.h"
//cpp stuff
namespace layers {    
   
    AveragePool::AveragePool(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/averagepool.spv");       
        dev = backend::g_device;
    }
       
        
    void AveragePool::init( std::vector<int> _kernel_shape,  std::string _auto_pad,  int _ceil_mode,  int _count_include_pad,  std::vector<int> _pads,  std::vector<int> _strides) {      
		 m_kernel_shape = _kernel_shape; 
 		 m_auto_pad = _auto_pad; 
 		 m_ceil_mode = _ceil_mode; 
 		 m_count_include_pad = _count_include_pad; 
 		 m_pads = _pads; 
 		 m_strides = _strides; 
  

    }
    
    void AveragePool::bind(std::string _X_i, std::string _Y_o){    
        m_X_i = _X_i; m_Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void AveragePool::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data);       
    }

    void AveragePool::forward(){ 
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data);
        //program->run();
    }

}

