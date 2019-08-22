#include "lstm.h"
//cpp stuff
namespace layers {    
   
    LSTM::LSTM(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\lstm.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* LSTM::_get_device() {
        
        return backend::device;
    }
    
    void LSTM::init( std::vector<float> _activation_alpha,  std::vector<float> _activation_beta,  std::vector<std::string> _activations,  float _clip,  std::string _direction,  int _hidden_size,  int _input_forget) {      
		 activation_alpha = _activation_alpha; 
 		 activation_beta = _activation_beta; 
 		 activations = _activations; 
 		 clip = _clip; 
 		 direction = _direction; 
 		 hidden_size = _hidden_size; 
 		 input_forget = _input_forget; 
  
    }
    
    void LSTM::bind(std::string _X_i, std::string _W_i, std::string _R_i, std::string _B_i, std::string _sequence_lens_i, std::string _initial_h_i, std::string _initial_c_i, std::string _P_i, std::string _Y_o, std::string _Y_h_o, std::string _Y_c_o){
        X_i = _X_i; W_i = _W_i; R_i = _R_i; B_i = _B_i; sequence_lens_i = _sequence_lens_i; initial_h_i = _initial_h_i; initial_c_i = _initial_c_i; P_i = _P_i; Y_o = _Y_o; Y_h_o = _Y_h_o; Y_c_o = _Y_c_o;

		//binding.X_i = tensor_dict[X_i]->shape();
  		//binding.W_i = tensor_dict[W_i]->shape();
  		//binding.R_i = tensor_dict[R_i]->shape();
  		//binding.B_i = tensor_dict[B_i]->shape();
  		//binding.sequence_lens_i = tensor_dict[sequence_lens_i]->shape();
  		//binding.initial_h_i = tensor_dict[initial_h_i]->shape();
  		//binding.initial_c_i = tensor_dict[initial_c_i]->shape();
  		//binding.P_i = tensor_dict[P_i]->shape();
 
		//binding.Y_o = tensor_dict[Y_o]->shape();
  		//binding.Y_h_o = tensor_dict[Y_h_o]->shape();
  		//binding.Y_c_o = tensor_dict[Y_c_o]->shape();
 
		//binding.activation_alpha = activation_alpha;
  		//binding.activation_beta = activation_beta;
  		//binding.activations = activations;
  		//binding.clip = clip;
  		//binding.direction = direction;
  		//binding.hidden_size = hidden_size;
  		//binding.input_forget = input_forget;
         
    }

    void LSTM::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[W_i]->data(), *tensor_dict[R_i]->data(), *tensor_dict[B_i]->data(), *tensor_dict[sequence_lens_i]->data(), *tensor_dict[initial_h_i]->data(), *tensor_dict[initial_c_i]->data(), *tensor_dict[P_i]->data(), *tensor_dict[Y_o]->data(), *tensor_dict[Y_h_o]->data(), *tensor_dict[Y_c_o]->data());
    }

}

