#include "randomnormal.h"
//cpp stuff
namespace layers {    
   
    RandomNormal::RandomNormal(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/randomnormal.spv");       
        dev = backend::g_device;
    }
       
        
    void RandomNormal::init( std::vector<int> _shape,  int _dtype,  float _mean,  float _scale,  float _seed) {      
		 m_shape = _shape; 
 		 m_dtype = _dtype; 
 		 m_mean = _mean; 
 		 m_scale = _scale; 
 		 m_seed = _seed; 
  

    }
    
    void RandomNormal::bind(std::string _output_o){    
        m_output_o = _output_o;        

		SHAPES.push_back(backend::tensor_dict[m_output_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void RandomNormal::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, 1);
        program->bind({0}, *_SHAPES, *backend::tensor_dict[m_output_o]->data);
    }

    void RandomNormal::forward(){ 
        program->run();
    }

}

