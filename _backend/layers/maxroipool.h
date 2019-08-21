#ifndef MAXROIPOOL_H
#define MAXROIPOOL_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

 ROI max pool consumes an input tensor X and region of interests (RoIs) to
 apply max pooling across each RoI, to produce output 4-D tensor of shape
 (num_rois, channels, pooled_shape[0], pooled_shape[1]).
input: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data.
input: RoIs (Regions of Interest) to pool over. Should be a 2-D tensor of shape (num_rois, 5) given as [[batch_id, x1, y1, x2, y2], ...].
output: RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_shape[0], pooled_shape[1]).
*/

//MaxRoiPool
//INPUTS:                   X_i, rois_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               pooled_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      spatial_scale
//OPTIONAL_PARAMETERS_TYPE: float


//class stuff
namespace backend {   

    class MaxRoiPool : public Layer {
        typedef struct {
            Shape_t pooled_shape; float spatial_scale;
			
            Shape_t X_i; Shape_t rois_i;
            
            Shape_t Y_o;
            
        } binding_descriptor;

        Shape_t pooled_shape; float spatial_scale;
        std::string X_i; std::string rois_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        MaxRoiPool(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( Shape_t _pooled_shape,  float _spatial_scale); 
        virtual void bind(std::string _X_i, std::string _rois_i, std::string _Y_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/maxroipool.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[rois_i]->data(), *tensor_dict[Y_o]->data());
        }

        ~MaxRoiPool() {}
    };
   
}
#endif

