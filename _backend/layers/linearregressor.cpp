#include "linearregressor.h"
//cpp stuff
namespace layers {    
   
    LinearRegressor::LinearRegressor(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/linearregressor.spv");       
        dev = backend::g_device;
    }
       
        
    void LinearRegressor::init( std::vector<float> _coefficients,  std::vector<float> _intercepts,  std::string _post_transform,  int _targets) {      
		 m_coefficients = _coefficients; 
 		 m_intercepts = _intercepts; 
 		 m_post_transform = _post_transform; 
 		 m_targets = _targets; 
  

    }
    
    void LinearRegressor::bind(std::string _X_i, std::string _Y_o){    
        m_X_i = _X_i; m_Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void LinearRegressor::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data);
    }

    void LinearRegressor::forward(){ 
        program->run();
    }

}

