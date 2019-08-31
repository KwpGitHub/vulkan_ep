#include "thresholdedrelu.h"
//cpp stuff
namespace layers {    
   
    ThresholdedRelu::ThresholdedRelu(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/thresholdedrelu.spv");       
        dev = backend::g_device;
    }
       
        
    void ThresholdedRelu::init( float _alpha) {      
		 m_alpha = _alpha; 
  

    }
    
    void ThresholdedRelu::bind(std::string _X_i, std::string _Y_o){    
        m_X_i = _X_i; m_Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void ThresholdedRelu::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_Y_o]->data);
    }

    void ThresholdedRelu::forward(){ 
        program->run();
    }

}

