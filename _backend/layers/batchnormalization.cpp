#include "batchnormalization.h"
//cpp stuff
namespace layers {    
   
    BatchNormalization::BatchNormalization(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/batchnormalization.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* BatchNormalization::_get_device() {        
        return backend::device;
    }
    
    void BatchNormalization::init( float _epsilon,  float _momentum) {      
		 epsilon = _epsilon; 
 		 momentum = _momentum; 
  
    }
    
    void BatchNormalization::bind(std::string _X_i, std::string _scale_i, std::string _B_i, std::string _mean_i, std::string _var_i, std::string _Y_o, std::string _mean_o, std::string _var_o, std::string _saved_mean_o, std::string _saved_var_o){
        X_i = _X_i; scale_i = _scale_i; B_i = _B_i; mean_i = _mean_i; var_i = _var_i; Y_o = _Y_o; mean_o = _mean_o; var_o = _var_o; saved_mean_o = _saved_mean_o; saved_var_o = _saved_var_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
  		binding.scale_i = backend::tensor_dict[scale_i]->shape();
  		binding.B_i = backend::tensor_dict[B_i]->shape();
  		binding.mean_i = backend::tensor_dict[mean_i]->shape();
  		binding.var_i = backend::tensor_dict[var_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
  		binding.mean_o = backend::tensor_dict[mean_o]->shape();
  		binding.var_o = backend::tensor_dict[var_o]->shape();
  		binding.saved_mean_o = backend::tensor_dict[saved_mean_o]->shape();
  		binding.saved_var_o = backend::tensor_dict[saved_var_o]->shape();
 
		//binding.epsilon = epsilon;
  		//binding.momentum = momentum;
         
    }

    void BatchNormalization::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[scale_i]->data(), *backend::tensor_dict[B_i]->data(), *backend::tensor_dict[mean_i]->data(), *backend::tensor_dict[var_i]->data(), *backend::tensor_dict[Y_o]->data(), *backend::tensor_dict[mean_o]->data(), *backend::tensor_dict[var_o]->data(), *backend::tensor_dict[saved_mean_o]->data(), *backend::tensor_dict[saved_var_o]->data());
    }

    void BatchNormalization::forward(){ 
        program->run();
    }

}

