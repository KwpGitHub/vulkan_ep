#include "featurevectorizer.h"
//cpp stuff
namespace layers {    
   
    FeatureVectorizer::FeatureVectorizer(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/featurevectorizer.spv");       
        dev = backend::g_device;
    }
       
        
    void FeatureVectorizer::init( std::vector<int> _inputdimensions) {      
		 m_inputdimensions = _inputdimensions; 
  

    }
    
    void FeatureVectorizer::bind(std::string _Y_o){    
        m_Y_o = _Y_o;        

		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void FeatureVectorizer::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_Y_o]->data);       
    }

    void FeatureVectorizer::forward(){ 
        (*program)({2, 1}, *_SHAPES, *backend::tensor_dict[m_Y_o]->data);
        //program->run();
    }

}

