#include "zipmap.h"
//cpp stuff
namespace layers {    
   
    ZipMap::ZipMap(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/zipmap.spv");       
        dev = backend::device;
    }
       
        
    void ZipMap::init( std::vector<int> _classlabels_int64s,  std::vector<std::string> _classlabels_strings) {      
		 classlabels_int64s = _classlabels_int64s; 
 		 classlabels_strings = _classlabels_strings; 
  

    }
    
    void ZipMap::bind(std::string _X_i, std::string _Z_o){    
        X_i = _X_i; Z_o = _Z_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Z_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void ZipMap::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[X_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[X_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[X_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[Z_o]->data);
    }

    void ZipMap::forward(){ 
        program->run();
    }

}

