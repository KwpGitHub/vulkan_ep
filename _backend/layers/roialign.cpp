#include "roialign.h"
//cpp stuff
namespace layers {    
   
    RoiAlign::RoiAlign(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/roialign.spv");       
        dev = backend::device;
    }
       
        
    void RoiAlign::init( std::string _mode,  int _output_height,  int _output_width,  int _sampling_ratio,  float _spatial_scale) {      
		 mode = _mode; 
 		 output_height = _output_height; 
 		 output_width = _output_width; 
 		 sampling_ratio = _sampling_ratio; 
 		 spatial_scale = _spatial_scale; 
  

    }
    
    void RoiAlign::bind(std::string _X_i, std::string _rois_i, std::string _batch_indices_i, std::string _Y_o){    
        X_i = _X_i; rois_i = _rois_i; batch_indices_i = _batch_indices_i; Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[rois_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[batch_indices_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void RoiAlign::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[X_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[X_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[X_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[X_i]->data, *backend::tensor_dict[rois_i]->data, *backend::tensor_dict[batch_indices_i]->data, *backend::tensor_dict[Y_o]->data);
    }

    void RoiAlign::forward(){ 
        program->run();
    }

}

