#include "lstm.h"
//cpp stuff
namespace layers {    
   
    LSTM::LSTM(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/lstm.spv");       
        dev = backend::device;
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
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[W_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[R_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[B_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[sequence_lens_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[initial_h_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[initial_c_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[P_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[Y_h_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[Y_c_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void LSTM::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[X_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[X_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[X_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[W_i]->data, *backend::tensor_dict[R_i]->data, *backend::tensor_dict[B_i]->data, *backend::tensor_dict[sequence_lens_i]->data, *backend::tensor_dict[initial_h_i]->data, *backend::tensor_dict[initial_c_i]->data, *backend::tensor_dict[P_i]->data, *backend::tensor_dict[Y_o]->data, *backend::tensor_dict[Y_h_o]->data, *backend::tensor_dict[Y_c_o]->data);
    }

    void LSTM::forward(){ 
        program->run();
    }

}

