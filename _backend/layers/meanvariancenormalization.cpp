#include "meanvariancenormalization.h"
//cpp stuff
namespace layers {    
   
    MeanVarianceNormalization::MeanVarianceNormalization(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/meanvariancenormalization.spv");       
        dev = backend::device;
    }
       
        
    void MeanVarianceNormalization::init( std::vector<int> _axes) {      
		 axes = _axes; 
  

    }
    
    void MeanVarianceNormalization::bind(std::string _X_i, std::string _Y_o){    
        X_i = _X_i; Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void MeanVarianceNormalization::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[X_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[X_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[X_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void MeanVarianceNormalization::forward(){ 
        program->run();
    }

}

