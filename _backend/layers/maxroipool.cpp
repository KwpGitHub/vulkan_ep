#include "maxroipool.h"
//cpp stuff
namespace layers {    
   
    MaxRoiPool::MaxRoiPool(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/maxroipool.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* MaxRoiPool::_get_device() {        
        return backend::device;
    }
    
    void MaxRoiPool::init( std::vector<int> _pooled_shape,  float _spatial_scale) {      
		 pooled_shape = _pooled_shape; 
 		 spatial_scale = _spatial_scale; 
  
    }
    
    void MaxRoiPool::bind(std::string _X_i, std::string _rois_i, std::string _Y_o){
        X_i = _X_i; rois_i = _rois_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
  		binding.rois_i = backend::tensor_dict[rois_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.pooled_shape = pooled_shape;
  		//binding.spatial_scale = spatial_scale;
         
    }

    void MaxRoiPool::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[rois_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void MaxRoiPool::forward(){ 
        //program->run();
    }

}

