#include "roialign.h"
//cpp stuff
namespace layers {    
   
    RoiAlign::RoiAlign(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/roialign.spv");       
        dev = backend::g_device;
    }
       
        
    void RoiAlign::init( std::string _mode,  int _output_height,  int _output_width,  int _sampling_ratio,  float _spatial_scale) {      
		 m_mode = _mode; 
 		 m_output_height = _output_height; 
 		 m_output_width = _output_width; 
 		 m_sampling_ratio = _sampling_ratio; 
 		 m_spatial_scale = _spatial_scale; 
  

    }
    
    void RoiAlign::bind(std::string _X_i, std::string _rois_i, std::string _batch_indices_i, std::string _Y_o){    
        m_X_i = _X_i; m_rois_i = _rois_i; m_batch_indices_i = _batch_indices_i; m_Y_o = _Y_o;        
		SHAPES.push_back(backend::tensor_dict[m_X_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_rois_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_batch_indices_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_Y_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void RoiAlign::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
        program->bind({2, 1}, *_SHAPES, *backend::tensor_dict[m_X_i]->data, *backend::tensor_dict[m_rois_i]->data, *backend::tensor_dict[m_batch_indices_i]->data, *backend::tensor_dict[m_Y_o]->data);
    }

    void RoiAlign::forward(){ 
        program->run();
    }

}

