#include "batchnormalization.h"
//cpp stuff
namespace layers {    
   
    BatchNormalization::BatchNormalization(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/batchnormalization.spv");       
        dev = backend::device;
    }
       
        
    void BatchNormalization::init( float _epsilon,  float _momentum) {      
		 epsilon = _epsilon; 
 		 momentum = _momentum; 
  

    }
    
    void BatchNormalization::bind(std::string _X_i, std::string _scale_i, std::string _B_i, std::string _mean_i, std::string _var_i, std::string _Y_o, std::string _mean_o, std::string _var_o, std::string _saved_mean_o, std::string _saved_var_o){    
        X_i = _X_i; scale_i = _scale_i; B_i = _B_i; mean_i = _mean_i; var_i = _var_i; Y_o = _Y_o; mean_o = _mean_o; var_o = _var_o; saved_mean_o = _saved_mean_o; saved_var_o = _saved_var_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[B_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[mean_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[var_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[mean_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[var_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[saved_mean_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[saved_var_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void BatchNormalization::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[scale_i]->data, *backend::tensor_dict[B_i]->data, *backend::tensor_dict[mean_i]->data, *backend::tensor_dict[var_i]->data, *backend::tensor_dict[Y_o]->data, *backend::tensor_dict[mean_o]->data, *backend::tensor_dict[var_o]->data, *backend::tensor_dict[saved_mean_o]->data, *backend::tensor_dict[saved_var_o]->data);
    }

    void BatchNormalization::forward(){ 
        program->run();
    }

}

