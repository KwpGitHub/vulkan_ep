#include "reducelogsumexp.h"
//cpp stuff
namespace layers {    
   
    ReduceLogSumExp::ReduceLogSumExp(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/reducelogsumexp.spv");       
        dev = backend::g_device;
    }
       
        
    void ReduceLogSumExp::init( std::vector<int> _axes,  int _keepdims) {      
		 m_axes = _axes; 
 		 m_keepdims = _keepdims; 
  

    }
    
    void ReduceLogSumExp::bind(std::string _data_i, std::string _reduced_o){    
        m_data_i = _data_i; m_reduced_o = _reduced_o;        
		SHAPES.push_back(backend::tensor_dict[m_data_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_reduced_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void ReduceLogSumExp::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE),
                        vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE));
        program->spec(SHAPES[0].w, SHAPES[0].h, SHAPES[0].d);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[m_data_i]->data, *backend::tensor_dict[m_reduced_o]->data);
    }

    void ReduceLogSumExp::forward(){ 
        program->run();
    }

}

