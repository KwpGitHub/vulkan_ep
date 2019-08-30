#include "svmclassifier.h"
//cpp stuff
namespace layers {    
   
    SVMClassifier::SVMClassifier(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/svmclassifier.spv");       
        dev = backend::g_device;
    }
       
        
    void SVMClassifier::init( std::vector<int> _classlabels_ints,  std::vector<std::string> _classlabels_strings,  std::vector<float> _coefficients,  std::vector<float> _kernel_params,  std::string _kernel_type,  std::string _post_transform,  std::vector<float> _prob_a,  std::vector<float> _prob_b,  std::vector<float> _rho,  std::vector<float> _support_vectors,  std::vector<int> _vectors_per_class) {      
		 m_classlabels_ints = _classlabels_ints; 
 		 m_classlabels_strings = _classlabels_strings; 
 		 m_coefficients = _coefficients; 
 		 m_kernel_params = _kernel_params; 
 		 m_kernel_type = _kernel_type; 
 		 m_post_transform = _post_transform; 
 		 m_prob_a = _prob_a; 
 		 m_prob_b = _prob_b; 
 		 m_rho = _rho; 
 		 m_support_vectors = _support_vectors; 
 		 m_vectors_per_class = _vectors_per_class; 
  

    }
    
    void SVMClassifier::bind(std::string _X_i, std::string _Y_o, std::string _Z_o){    
        m_X_i = _X_i; m_Y_o = _Y_o; m_Z_o = _Z_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_Z_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void SVMClassifier::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(SHAPES[0].w, SHAPES[0].h, SHAPES[0].d);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data, *backend::tensor_dict[m_Z_o]->data);
    }

    void SVMClassifier::forward(){ 
        program->run();
    }

}

