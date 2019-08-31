#include "linearclassifier.h"
//cpp stuff
namespace layers {    
   
    LinearClassifier::LinearClassifier(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/linearclassifier.spv");       
        dev = backend::g_device;
    }
       
        
    void LinearClassifier::init( std::vector<float> _coefficients,  std::vector<int> _classlabels_ints,  std::vector<std::string> _classlabels_strings,  std::vector<float> _intercepts,  int _multi_class,  std::string _post_transform) {      
		 m_coefficients = _coefficients; 
 		 m_classlabels_ints = _classlabels_ints; 
 		 m_classlabels_strings = _classlabels_strings; 
 		 m_intercepts = _intercepts; 
 		 m_multi_class = _multi_class; 
 		 m_post_transform = _post_transform; 
  

    }
    
    void LinearClassifier::bind(std::string _X_i, std::string _Y_o, std::string _Z_o){    
        m_X_i = _X_i; m_Y_o = _Y_o; m_Z_o = _Z_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_Z_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void LinearClassifier::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data, *backend::tensor_dict[m_Z_o]->data);
    }

    void LinearClassifier::forward(){ 
        program->run();
    }

}

