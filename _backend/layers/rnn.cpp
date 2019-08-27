#include "rnn.h"
//cpp stuff
namespace layers {    
   
    RNN::RNN(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/rnn.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* RNN::_get_device() {        
        return backend::device;
    }
    
    void RNN::init( std::vector<float> _activation_alpha,  std::vector<float> _activation_beta,  std::vector<std::string> _activations,  float _clip,  std::string _direction,  int _hidden_size) {      
		 activation_alpha = _activation_alpha; 
 		 activation_beta = _activation_beta; 
 		 activations = _activations; 
 		 clip = _clip; 
 		 direction = _direction; 
 		 hidden_size = _hidden_size; 
  
    }
    
    void RNN::bind(std::string _X_i, std::string _W_i, std::string _R_i, std::string _B_i, std::string _sequence_lens_i, std::string _initial_h_i, std::string _Y_o, std::string _Y_h_o){
        X_i = _X_i; W_i = _W_i; R_i = _R_i; B_i = _B_i; sequence_lens_i = _sequence_lens_i; initial_h_i = _initial_h_i; Y_o = _Y_o; Y_h_o = _Y_h_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
  		binding.W_i = backend::tensor_dict[W_i]->shape();
  		binding.R_i = backend::tensor_dict[R_i]->shape();
  		binding.B_i = backend::tensor_dict[B_i]->shape();
  		binding.sequence_lens_i = backend::tensor_dict[sequence_lens_i]->shape();
  		binding.initial_h_i = backend::tensor_dict[initial_h_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
  		binding.Y_h_o = backend::tensor_dict[Y_h_o]->shape();
 
		//binding.activation_alpha = activation_alpha;
  		//binding.activation_beta = activation_beta;
  		//binding.activations = activations;
  		//binding.clip = clip;
  		//binding.direction = direction;
  		//binding.hidden_size = hidden_size;
         
    }

    void RNN::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[W_i]->data(), *backend::tensor_dict[R_i]->data(), *backend::tensor_dict[B_i]->data(), *backend::tensor_dict[sequence_lens_i]->data(), *backend::tensor_dict[initial_h_i]->data(), *backend::tensor_dict[Y_o]->data(), *backend::tensor_dict[Y_h_o]->data());
    }

    void RNN::forward(){ 
        program->run();
    }

}

