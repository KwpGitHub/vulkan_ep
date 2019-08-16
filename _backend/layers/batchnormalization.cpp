#include "BatchNormalization.h"

//cpp stuff
namespace backend {    
   
    BatchNormalization::BatchNormalization(std::string n) : Layer(n) { }
       
    vuh::Device* BatchNormalization::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void BatchNormalization::init( float _epsilon,  float _momentum) {      
		 epsilon = _epsilon; 
 		 momentum = _momentum; 
  
    }
    
    void BatchNormalization::bind(std::string _X_input, std::string _scale_input, std::string _B_input, std::string _mean_input, std::string _var_input, std::string _Y_output, std::string _mean_output_opt, std::string _var_output_opt, std::string _saved_mean_output_opt, std::string _saved_var_output_opt){
        X_input = _X_input; scale_input = _scale_input; B_input = _B_input; mean_input = _mean_input; var_input = _var_input; Y_output = _Y_output; mean_output_opt = _mean_output_opt; var_output_opt = _var_output_opt; saved_mean_output_opt = _saved_mean_output_opt; saved_var_output_opt = _saved_var_output_opt;
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.scale_input = tensor_dict[scale_input]->shape();
  		binding.B_input = tensor_dict[B_input]->shape();
  		binding.mean_input = tensor_dict[mean_input]->shape();
  		binding.var_input = tensor_dict[var_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
  		binding.mean_output_opt = tensor_dict[mean_output_opt]->shape();
  		binding.var_output_opt = tensor_dict[var_output_opt]->shape();
  		binding.saved_mean_output_opt = tensor_dict[saved_mean_output_opt]->shape();
  		binding.saved_var_output_opt = tensor_dict[saved_var_output_opt]->shape();
 
		binding.epsilon = epsilon;
  		binding.momentum = momentum;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/batchnormalization.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[scale_input]->data(), *tensor_dict[B_input]->data(), *tensor_dict[mean_input]->data(), *tensor_dict[var_input]->data(), *tensor_dict[Y_output]->data(), *tensor_dict[mean_output_opt]->data(), *tensor_dict[var_output_opt]->data(), *tensor_dict[saved_mean_output_opt]->data(), *tensor_dict[saved_var_output_opt]->data());
    }
    
}

    //backend::nn;

//python stuff


