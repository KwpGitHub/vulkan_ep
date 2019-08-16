#include "RNN.h"

//cpp stuff
namespace backend {    
   
    RNN::RNN(std::string n) : Layer(n) { }
       
    vuh::Device* RNN::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void RNN::init( float _clip,  int _direction,  int _hidden_size) {      
		 clip = _clip; 
 		 direction = _direction; 
 		 hidden_size = _hidden_size; 
  
    }
    
    void RNN::bind(std::string _activation_alpha, std::string _activation_beta, std::string _activations, std::string _X_input, std::string _W_input, std::string _R_input, std::string _B_input_opt, std::string _sequence_lens_input_opt, std::string _initial_h_input_opt, std::string _Y_output_opt, std::string _Y_h_output_opt){
        activation_alpha = _activation_alpha; activation_beta = _activation_beta; activations = _activations; X_input = _X_input; W_input = _W_input; R_input = _R_input; B_input_opt = _B_input_opt; sequence_lens_input_opt = _sequence_lens_input_opt; initial_h_input_opt = _initial_h_input_opt; Y_output_opt = _Y_output_opt; Y_h_output_opt = _Y_h_output_opt;
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.W_input = tensor_dict[W_input]->shape();
  		binding.R_input = tensor_dict[R_input]->shape();
  		binding.B_input_opt = tensor_dict[B_input_opt]->shape();
  		binding.sequence_lens_input_opt = tensor_dict[sequence_lens_input_opt]->shape();
  		binding.initial_h_input_opt = tensor_dict[initial_h_input_opt]->shape();
 
		binding.Y_output_opt = tensor_dict[Y_output_opt]->shape();
  		binding.Y_h_output_opt = tensor_dict[Y_h_output_opt]->shape();
 
		binding.clip = clip;
  		binding.direction = direction;
  		binding.hidden_size = hidden_size;
 
		binding.activation_alpha = tensor_dict[activation_alpha]->shape();
  		binding.activation_beta = tensor_dict[activation_beta]->shape();
  		binding.activations = tensor_dict[activations]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/rnn.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[activation_alpha]->data(), *tensor_dict[activation_beta]->data(), *tensor_dict[activations]->data(), *tensor_dict[X_input]->data(), *tensor_dict[W_input]->data(), *tensor_dict[R_input]->data(), *tensor_dict[B_input_opt]->data(), *tensor_dict[sequence_lens_input_opt]->data(), *tensor_dict[initial_h_input_opt]->data(), *tensor_dict[Y_output_opt]->data(), *tensor_dict[Y_h_output_opt]->data());
    }
    
}

    //backend::nn;

//python stuff


