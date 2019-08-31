#include "imputer.h"
//cpp stuff
namespace layers {    
   
    Imputer::Imputer(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/imputer.spv");       
        dev = backend::g_device;
    }
       
        
    void Imputer::init( std::vector<float> _imputed_value_floats,  std::vector<int> _imputed_value_int64s,  float _replaced_value_float,  int _replaced_value_int64) {      
		 m_imputed_value_floats = _imputed_value_floats; 
 		 m_imputed_value_int64s = _imputed_value_int64s; 
 		 m_replaced_value_float = _replaced_value_float; 
 		 m_replaced_value_int64 = _replaced_value_int64; 
  

    }
    
    void Imputer::bind(std::string _X_i, std::string _Y_o){    
        m_X_i = _X_i; m_Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Imputer::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data);
    }

    void Imputer::forward(){ 
        program->run();
    }

}

