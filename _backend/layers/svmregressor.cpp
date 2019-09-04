#include "svmregressor.h"
//cpp stuff
namespace layers {    
   
    SVMRegressor::SVMRegressor(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/svmregressor.spv");       
        dev = backend::g_device;
    }
       
        
    void SVMRegressor::init( std::vector<float> _coefficients,  std::vector<float> _kernel_params,  std::string _kernel_type,  int _n_supports,  int _one_class,  std::string _post_transform,  std::vector<float> _rho,  std::vector<float> _support_vectors) {      
		 m_coefficients = _coefficients; 
 		 m_kernel_params = _kernel_params; 
 		 m_kernel_type = _kernel_type; 
 		 m_n_supports = _n_supports; 
 		 m_one_class = _one_class; 
 		 m_post_transform = _post_transform; 
 		 m_rho = _rho; 
 		 m_support_vectors = _support_vectors; 
  

    }
    
    void SVMRegressor::bind(std::string _X_i, std::string _Y_o){    
        m_X_i = _X_i; m_Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void SVMRegressor::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data);       
    }

    void SVMRegressor::forward(){ 
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data);
        //program->run();
    }

}

