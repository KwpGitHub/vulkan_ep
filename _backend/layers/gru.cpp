#include "gru.h"
//cpp stuff
namespace layers {    
   
    GRU::GRU(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/gru.spv");       
        dev = backend::device;
    }
       
        
    void GRU::init( std::vector<float> _activation_alpha,  std::vector<float> _activation_beta,  std::vector<std::string> _activations,  float _clip,  std::string _direction,  int _hidden_size,  int _linear_before_reset) {      
		 activation_alpha = _activation_alpha; 
 		 activation_beta = _activation_beta; 
 		 activations = _activations; 
 		 clip = _clip; 
 		 direction = _direction; 
 		 hidden_size = _hidden_size; 
 		 linear_before_reset = _linear_before_reset; 
  

    }
    
    void GRU::bind(std::string _X_i, std::string _W_i, std::string _R_i, std::string _B_i, std::string _sequence_lens_i, std::string _initial_h_i, std::string _Y_o, std::string _Y_h_o){    
        X_i = _X_i; W_i = _W_i; R_i = _R_i; B_i = _B_i; sequence_lens_i = _sequence_lens_i; initial_h_i = _initial_h_i; Y_o = _Y_o; Y_h_o = _Y_h_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[W_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[R_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[B_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[sequence_lens_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[initial_h_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[Y_h_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void GRU::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128, 0.1f}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[W_i]->data, *backend::tensor_dict[R_i]->data, *backend::tensor_dict[B_i]->data, *backend::tensor_dict[sequence_lens_i]->data, *backend::tensor_dict[initial_h_i]->data, *backend::tensor_dict[Y_o]->data, *backend::tensor_dict[Y_h_o]->data);
    }

    void GRU::forward(){ 
        program->run();
    }

}

