#include "MaxRoiPool.h"
//cpp stuff
namespace backend {    
   
    MaxRoiPool::MaxRoiPool(std::string name) : Layer(name) { }
       
    vuh::Device* MaxRoiPool::_get_device() {
        
        return device;
    }
    
    void MaxRoiPool::init( Shape_t _pooled_shape,  float _spatial_scale) {      
		 pooled_shape = _pooled_shape; 
 		 spatial_scale = _spatial_scale; 
  
    }
    
    void MaxRoiPool::bind(std::string _X_i, std::string _rois_i, std::string _Y_o){
        X_i = _X_i; rois_i = _rois_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
  		binding.rois_i = tensor_dict[rois_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.pooled_shape = pooled_shape;
  		binding.spatial_scale = spatial_scale;
 

        
    }
}

