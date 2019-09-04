#include "randomuniform.h"
//cpp stuff
namespace layers {    
   
    RandomUniform::RandomUniform(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/randomuniform.spv");       
        dev = backend::g_device;
    }
       
        
    void RandomUniform::init( std::vector<int> _shape,  int _dtype,  float _high,  float _low,  float _seed) {      
		 m_shape = _shape; 
 		 m_dtype = _dtype; 
 		 m_high = _high; 
 		 m_low = _low; 
 		 m_seed = _seed; 
  

    }
    
    void RandomUniform::bind(std::string _output_o){    
        m_output_o = _output_o;        

		SHAPES.push_back(backend::tensor_dict[m_output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void RandomUniform::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
       
    }

    void RandomUniform::forward(){ 
        program->operator()({2, 1}, *_SHAPES, *backend::tensor_dict[m_output_o]->data);
        //program->run();
    }

}

