#include "../layer.h"
#ifndef ROIALIGN_H
#define ROIALIGN_H 
/*

Region of Interest (RoI) align operation described in the
[Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
RoiAlign consumes an input tensor X and region of interests (rois)
to apply pooling across each RoI; it produces a 4-D tensor of shape
(num_rois, C, output_height, output_width).

RoiAlign is proposed to avoid the misalignment by removing
quantizations while converting from original image into feature
map and from feature map into RoI feature; in each ROI bin,
the value of the sampled locations are computed directly
through bilinear interpolation.

input: Input data tensor from the previous operator; 4-D feature map of shape (N, C, H, W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data.
input: RoIs (Regions of Interest) to pool over; rois is 2-D input of shape (num_rois, 4) given as [[x1, y1, x2, y2], ...]. The RoIs' coordinates are in the coordinate system of the input image. Each coordinate set has a 1:1 correspondence with the 'batch_indices' input.
input: 1-D tensor of shape (num_rois,) with each element denoting the index of the corresponding image in the batch.
output: RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width). The r-th batch element Y[r-1] is a pooled feature map corresponding to the r-th RoI X[r-1].
//*/
//RoiAlign
//INPUTS:                   X_input, rois_input, batch_indices_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      mode, output_height, output_width, sampling_ratio, spatial_scale
//OPTIONAL_PARAMETERS_TYPE: int, int, int, int, float

//class stuff
namespace backend {   

    class RoiAlign : public Layer {
        typedef struct {
            int mode; int output_height; int output_width; int sampling_ratio; float spatial_scale;
			
            Shape_t X_input; Shape_t rois_input; Shape_t batch_indices_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int mode; int output_height; int output_width; int sampling_ratio; float spatial_scale;
        std::string X_input; std::string rois_input; std::string batch_indices_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        RoiAlign(std::string n, int mode, int output_height, int output_width, int sampling_ratio, float spatial_scale);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string rois_input, std::string batch_indices_input, std::string Y_output); 

        ~RoiAlign() {}

    };
    
}

#endif

