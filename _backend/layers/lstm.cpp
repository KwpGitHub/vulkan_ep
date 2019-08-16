#include "LSTM.h"

//cpp stuff
namespace backend {    
   
    LSTM::LSTM(std::string n, float clip, int direction, int hidden_size, int input_forget) : Layer(n) { }
       
    vuh::Device* LSTM::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void LSTM::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.W_input = tensor_dict[W_input]->shape();
  		binding.R_input = tensor_dict[R_input]->shape();
  		binding.B_input_opt = tensor_dict[B_input_opt]->shape();
  		binding.sequence_lens_input_opt = tensor_dict[sequence_lens_input_opt]->shape();
  		binding.initial_h_input_opt = tensor_dict[initial_h_input_opt]->shape();
  		binding.initial_c_input_opt = tensor_dict[initial_c_input_opt]->shape();
  		binding.P_input_opt = tensor_dict[P_input_opt]->shape();
 
		binding.Y_output_opt = tensor_dict[Y_output_opt]->shape();
  		binding.Y_h_output_opt = tensor_dict[Y_h_output_opt]->shape();
  		binding.Y_c_output_opt = tensor_dict[Y_c_output_opt]->shape();
 
		binding.clip = clip;
  		binding.direction = direction;
  		binding.hidden_size = hidden_size;
  		binding.input_forget = input_forget;
  		binding.activation_alpha = tensor_dict[activation_alpha]->shape();
  		binding.activation_beta = tensor_dict[activation_beta]->shape();
  		binding.activations = tensor_dict[activations]->shape();
 
    }
    
    void LSTM::call(std::string activation_alpha, std::string activation_beta, std::string activations, std::string X_input, std::string W_input, std::string R_input, std::string B_input_opt, std::string sequence_lens_input_opt, std::string initial_h_input_opt, std::string initial_c_input_opt, std::string P_input_opt, std::string Y_output_opt, std::string Y_h_output_opt, std::string Y_c_output_opt){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/lstm.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[activation_alpha]->data(), *tensor_dict[activation_beta]->data(), *tensor_dict[activations]->data(), *tensor_dict[X_input]->data(), *tensor_dict[W_input]->data(), *tensor_dict[R_input]->data(), *tensor_dict[B_input_opt]->data(), *tensor_dict[sequence_lens_input_opt]->data(), *tensor_dict[initial_h_input_opt]->data(), *tensor_dict[initial_c_input_opt]->data(), *tensor_dict[P_input_opt]->data(), *tensor_dict[Y_output_opt]->data(), *tensor_dict[Y_h_output_opt]->data(), *tensor_dict[Y_c_output_opt]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


