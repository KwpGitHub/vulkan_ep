#include "maxunpool.h"
//cpp stuff
namespace layers {    
   
    MaxUnpool::MaxUnpool(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/maxunpool.spv");       
        dev = backend::g_device;
    }
       
        
    void MaxUnpool::init( std::vector<int> _kernel_shape,  std::vector<int> _pads,  std::vector<int> _strides) {      
		 m_kernel_shape = _kernel_shape; 
 		 m_pads = _pads; 
 		 m_strides = _strides; 
  

    }
    
    void MaxUnpool::bind(std::string _X_i, std::string _I_i, std::string _output_shape_i, std::string _output_o){    
        m_X_i = _X_i; m_I_i = _I_i; m_output_shape_i = _output_shape_i; m_output_o = _output_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_I_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_output_shape_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void MaxUnpool::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        program->bind({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_I_i]->data, *backend::tensor_dict[m_output_shape_i]->data, *backend::tensor_dict[m_output_o]->data);
    }

    void MaxUnpool::forward(){ 
        program->run();
    }

}

