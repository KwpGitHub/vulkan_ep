#include "featurevectorizer.h"
//cpp stuff
namespace layers {    
   
    FeatureVectorizer::FeatureVectorizer(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/featurevectorizer.spv");       
        dev = backend::device;
    }
       
        
    void FeatureVectorizer::init( std::vector<int> _inputdimensions) {      
		 inputdimensions = _inputdimensions; 
  

    }
    
    void FeatureVectorizer::bind(std::string _Y_o){    
        Y_o = _Y_o;        

		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void FeatureVectorizer::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[Y_o]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[Y_o]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[Y_o]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[Y_o]->data);
    }

    void FeatureVectorizer::forward(){ 
        program->run();
    }

}

