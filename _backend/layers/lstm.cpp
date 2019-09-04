#include "lstm.h"
//cpp stuff
namespace layers {    
   
    LSTM::LSTM(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/lstm.spv");       
        dev = backend::g_device;
    }
       
        
    void LSTM::init( std::vector<float> _activation_alpha,  std::vector<float> _activation_beta,  std::vector<std::string> _activations,  float _clip,  std::string _direction,  int _hidden_size,  int _input_forget) {      
		 m_activation_alpha = _activation_alpha; 
 		 m_activation_beta = _activation_beta; 
 		 m_activations = _activations; 
 		 m_clip = _clip; 
 		 m_direction = _direction; 
 		 m_hidden_size = _hidden_size; 
 		 m_input_forget = _input_forget; 
  

    }
    
    void LSTM::bind(std::string _X_i, std::string _W_i, std::string _R_i, std::string _B_i, std::string _sequence_lens_i, std::string _initial_h_i, std::string _initial_c_i, std::string _P_i, std::string _Y_o, std::string _Y_h_o, std::string _Y_c_o){    
        m_X_i = _X_i; m_W_i = _W_i; m_R_i = _R_i; m_B_i = _B_i; m_sequence_lens_i = _sequence_lens_i; m_initial_h_i = _initial_h_i; m_initial_c_i = _initial_c_i; m_P_i = _P_i; m_Y_o = _Y_o; m_Y_h_o = _Y_h_o; m_Y_c_o = _Y_c_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_W_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_R_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_B_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_sequence_lens_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_initial_h_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_initial_c_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_P_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_Y_h_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_Y_c_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void LSTM::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_W_i]->data, *backend::tensor_dict[m_R_i]->data, *backend::tensor_dict[m_B_i]->data, *backend::tensor_dict[m_sequence_lens_i]->data, *backend::tensor_dict[m_initial_h_i]->data, *backend::tensor_dict[m_initial_c_i]->data, *backend::tensor_dict[m_P_i]->data, *backend::tensor_dict[m_Y_o]->data, *backend::tensor_dict[m_Y_h_o]->data, *backend::tensor_dict[m_Y_c_o]->data);       
    }

    void LSTM::forward(){ 
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_W_i]->data, *backend::tensor_dict[m_R_i]->data, *backend::tensor_dict[m_B_i]->data, *backend::tensor_dict[m_sequence_lens_i]->data, *backend::tensor_dict[m_initial_h_i]->data, *backend::tensor_dict[m_initial_c_i]->data, *backend::tensor_dict[m_P_i]->data, *backend::tensor_dict[m_Y_o]->data, *backend::tensor_dict[m_Y_h_o]->data, *backend::tensor_dict[m_Y_c_o]->data);
        //program->run();
    }

}

