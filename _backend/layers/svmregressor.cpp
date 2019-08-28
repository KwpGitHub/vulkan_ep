#include "svmregressor.h"
//cpp stuff
namespace layers {    
   
    SVMRegressor::SVMRegressor(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/svmregressor.spv");       
        dev = backend::device;
    }
       
        
    void SVMRegressor::init( std::vector<float> _coefficients,  std::vector<float> _kernel_params,  std::string _kernel_type,  int _n_supports,  int _one_class,  std::string _post_transform,  std::vector<float> _rho,  std::vector<float> _support_vectors) {      
		 coefficients = _coefficients; 
 		 kernel_params = _kernel_params; 
 		 kernel_type = _kernel_type; 
 		 n_supports = _n_supports; 
 		 one_class = _one_class; 
 		 post_transform = _post_transform; 
 		 rho = _rho; 
 		 support_vectors = _support_vectors; 
  

    }
    
    void SVMRegressor::bind(std::string _X_i, std::string _Y_o){    
        X_i = _X_i; Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void SVMRegressor::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void SVMRegressor::forward(){ 
        program->run();
    }

}

