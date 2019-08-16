#include "../layer.h"
#ifndef MAXROIPOOL_H
#define MAXROIPOOL_H 
/*

 ROI max pool consumes an input tensor X and region of interests (RoIs) to
 apply max pooling across each RoI, to produce output 4-D tensor of shape
 (num_rois, channels, pooled_shape[0], pooled_shape[1]).
input: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data.
input: RoIs (Regions of Interest) to pool over. Should be a 2-D tensor of shape (num_rois, 5) given as [[batch_id, x1, y1, x2, y2], ...].
output: RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_shape[0], pooled_shape[1]).
//*/
//MaxRoiPool
//INPUTS:                   X_input, rois_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
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
			
            Shape_t X_input; Shape_t rois_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        Shape_t pooled_shape; float spatial_scale;
        std::string X_input; std::string rois_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        MaxRoiPool(std::string n);
    
        void forward() { program->run(); }
        
        void init( Shape_t _pooled_shape,  float _spatial_scale); 
        void bind(std::string _X_input, std::string _rois_input, std::string _Y_output); 

        ~MaxRoiPool() {}

    };
    
}

#endif

