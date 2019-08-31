#include "batchnormalization.h"
//cpp stuff
namespace layers {    
   
    BatchNormalization::BatchNormalization(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/batchnormalization.spv");       
        dev = backend::g_device;
    }
       
        
    void BatchNormalization::init( float _epsilon,  float _momentum) {      
		 m_epsilon = _epsilon; 
 		 m_momentum = _momentum; 
  

    }
    
    void BatchNormalization::bind(std::string _X_i, std::string _scale_i, std::string _B_i, std::string _mean_i, std::string _var_i, std::string _Y_o, std::string _mean_o, std::string _var_o, std::string _saved_mean_o, std::string _saved_var_o){    
        m_X_i = _X_i; m_scale_i = _scale_i; m_B_i = _B_i; m_mean_i = _mean_i; m_var_i = _var_i; m_Y_o = _Y_o; m_mean_o = _mean_o; m_var_o = _var_o; m_saved_mean_o = _saved_mean_o; m_saved_var_o = _saved_var_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_scale_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_B_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_mean_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_var_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_mean_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_var_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_saved_mean_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_saved_var_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void BatchNormalization::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_scale_i]->data, *backend::tensor_dict[m_B_i]->data, *backend::tensor_dict[m_mean_i]->data, *backend::tensor_dict[m_var_i]->data, *backend::tensor_dict[m_Y_o]->data, *backend::tensor_dict[m_mean_o]->data, *backend::tensor_dict[m_var_o]->data, *backend::tensor_dict[m_saved_mean_o]->data, *backend::tensor_dict[m_saved_var_o]->data);
    }

    void BatchNormalization::forward(){ 
        program->run();
    }

}

