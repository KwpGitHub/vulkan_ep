#include "svmclassifier.h"
//cpp stuff
namespace layers {    
   
    SVMClassifier::SVMClassifier(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/svmclassifier.spv");       
        dev = backend::device;
    }
       
        
    void SVMClassifier::init( std::vector<int> _classlabels_ints,  std::vector<std::string> _classlabels_strings,  std::vector<float> _coefficients,  std::vector<float> _kernel_params,  std::string _kernel_type,  std::string _post_transform,  std::vector<float> _prob_a,  std::vector<float> _prob_b,  std::vector<float> _rho,  std::vector<float> _support_vectors,  std::vector<int> _vectors_per_class) {      
		 classlabels_ints = _classlabels_ints; 
 		 classlabels_strings = _classlabels_strings; 
 		 coefficients = _coefficients; 
 		 kernel_params = _kernel_params; 
 		 kernel_type = _kernel_type; 
 		 post_transform = _post_transform; 
 		 prob_a = _prob_a; 
 		 prob_b = _prob_b; 
 		 rho = _rho; 
 		 support_vectors = _support_vectors; 
 		 vectors_per_class = _vectors_per_class; 
  

    }
    
    void SVMClassifier::bind(std::string _X_i, std::string _Y_o, std::string _Z_o){    
        X_i = _X_i; Y_o = _Y_o; Z_o = _Z_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[Z_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void SVMClassifier::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[X_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[X_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[X_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Y_o]->data, *backend::tensor_dict[Z_o]->data);
    }

    void SVMClassifier::forward(){ 
        program->run();
    }

}

