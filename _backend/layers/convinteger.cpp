#include "convinteger.h"
//cpp stuff
namespace layers {    
   
    ConvInteger::ConvInteger(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/convinteger.spv");       
        dev = backend::device;
    }
       
        
    void ConvInteger::init( std::string _auto_pad,  std::vector<int> _dilations,  int _group,  std::vector<int> _kernel_shape,  std::vector<int> _pads,  std::vector<int> _strides) {      
		 auto_pad = _auto_pad; 
 		 dilations = _dilations; 
 		 group = _group; 
 		 kernel_shape = _kernel_shape; 
 		 pads = _pads; 
 		 strides = _strides; 
  

    }
    
    void ConvInteger::bind(std::string _x_i, std::string _w_i, std::string _x_zero_point_i, std::string _w_zero_point_i, std::string _y_o){    
        x_i = _x_i; w_i = _w_i; x_zero_point_i = _x_zero_point_i; w_zero_point_i = _w_zero_point_i; y_o = _y_o;        
		SHAPES.push_back(backend::tensor_dict[x_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[w_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[x_zero_point_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[w_zero_point_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void ConvInteger::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[x_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[x_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[x_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[x_i]->data, *backend::tensor_dict[w_i]->data, *backend::tensor_dict[x_zero_point_i]->data, *backend::tensor_dict[w_zero_point_i]->data, *backend::tensor_dict[y_o]->data);
    }

    void ConvInteger::forward(){ 
        program->run();
    }

}

