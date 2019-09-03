#include "topk.h"
//cpp stuff
namespace layers {    
   
    TopK::TopK(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/topk.spv");       
        dev = backend::g_device;
    }
       
        
    void TopK::init( int _axis) {      
		 m_axis = _axis; 
  

    }
    
    void TopK::bind(std::string _X_i, std::string _K_i, std::string _Values_o, std::string _Indices_o){    
        m_X_i = _X_i; m_K_i = _K_i; m_Values_o = _Values_o; m_Indices_o = _Indices_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_K_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Values_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_Indices_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void TopK::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        program->bind({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_K_i]->data, *backend::tensor_dict[m_Values_o]->data, *backend::tensor_dict[m_Indices_o]->data);
    }

    void TopK::forward(){ 
        program->run();
    }

}

