#include "GRU.h"
//cpp stuff
namespace backend {    
   
    GRU::GRU(std::string name) : Layer(name) { }
       
    vuh::Device* GRU::_get_device() {
        
        return device;
    }
    
    void GRU::init( float _clip,  int _direction,  int _hidden_size,  int _linear_before_reset) {      
		 clip = _clip; 
 		 direction = _direction; 
 		 hidden_size = _hidden_size; 
 		 linear_before_reset = _linear_before_reset; 
  
    }
    
    void GRU::bind(std::string _activation_alpha, std::string _activation_beta, std::string _activations, std::string _X_i, std::string _W_i, std::string _R_i, std::string _B_i, std::string _sequence_lens_i, std::string _initial_h_i, std::string _Y_o, std::string _Y_h_o){
        activation_alpha = _activation_alpha; activation_beta = _activation_beta; activations = _activations; X_i = _X_i; W_i = _W_i; R_i = _R_i; B_i = _B_i; sequence_lens_i = _sequence_lens_i; initial_h_i = _initial_h_i; Y_o = _Y_o; Y_h_o = _Y_h_o;

		binding.X_i = tensor_dict[X_i]->shape();
  		binding.W_i = tensor_dict[W_i]->shape();
  		binding.R_i = tensor_dict[R_i]->shape();
  		binding.B_i = tensor_dict[B_i]->shape();
  		binding.sequence_lens_i = tensor_dict[sequence_lens_i]->shape();
  		binding.initial_h_i = tensor_dict[initial_h_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
  		binding.Y_h_o = tensor_dict[Y_h_o]->shape();
 
		binding.clip = clip;
  		binding.direction = direction;
  		binding.hidden_size = hidden_size;
  		binding.linear_before_reset = linear_before_reset;
 
		binding.activation_alpha = tensor_dict[activation_alpha]->shape();
  		binding.activation_beta = tensor_dict[activation_beta]->shape();
  		binding.activations = tensor_dict[activations]->shape();
 
        
    }
}

