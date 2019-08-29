#include "convtranspose.h"
//cpp stuff
namespace layers {    
   
    ConvTranspose::ConvTranspose(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/convtranspose.spv");       
        dev = backend::device;
    }
       
        
    void ConvTranspose::init( std::string _auto_pad,  std::vector<int> _dilations,  int _group,  std::vector<int> _kernel_shape,  std::vector<int> _output_padding,  std::vector<int> _output_shape,  std::vector<int> _pads,  std::vector<int> _strides) {      
		 auto_pad = _auto_pad; 
 		 dilations = _dilations; 
 		 group = _group; 
 		 kernel_shape = _kernel_shape; 
 		 output_padding = _output_padding; 
 		 output_shape = _output_shape; 
 		 pads = _pads; 
 		 strides = _strides; 
  

    }
    
    void ConvTranspose::bind(std::string _X_i, std::string _W_i, std::string _B_i, std::string _Y_o){    
        X_i = _X_i; W_i = _W_i; B_i = _B_i; Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[W_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[B_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void ConvTranspose::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[X_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[X_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[X_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[W_i]->data, *backend::tensor_dict[B_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void ConvTranspose::forward(){ 
        program->run();
    }

}

