#include "imputer.h"
//cpp stuff
namespace layers {    
   
    Imputer::Imputer(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/imputer.spv");       
        dev = backend::device;
    }
       
        
    void Imputer::init( std::vector<float> _imputed_value_floats,  std::vector<int> _imputed_value_int64s,  float _replaced_value_float,  int _replaced_value_int64) {      
		 imputed_value_floats = _imputed_value_floats; 
 		 imputed_value_int64s = _imputed_value_int64s; 
 		 replaced_value_float = _replaced_value_float; 
 		 replaced_value_int64 = _replaced_value_int64; 
  

    }
    
    void Imputer::bind(std::string _X_i, std::string _Y_o){    
        X_i = _X_i; Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void Imputer::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[X_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[X_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[X_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void Imputer::forward(){ 
        program->run();
    }

}

