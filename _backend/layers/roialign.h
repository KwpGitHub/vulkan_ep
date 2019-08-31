#ifndef ROIALIGN_H
#define ROIALIGN_H 

#include "../layer.h"

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
*/

//RoiAlign
//INPUTS:                   X_i, rois_i, batch_indices_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      mode, output_height, output_width, sampling_ratio, spatial_scale
//OPTIONAL_PARAMETERS_TYPE: std::string, int, int, int, float


//class stuff
namespace layers {   

    class RoiAlign : public backend::Layer {
        typedef struct {
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::string m_mode; int m_output_height; int m_output_width; int m_sampling_ratio; float m_spatial_scale;
        std::string m_X_i; std::string m_rois_i; std::string m_batch_indices_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        RoiAlign(std::string name);
        
        virtual void forward();        
        virtual void init( std::string _mode,  int _output_height,  int _output_width,  int _sampling_ratio,  float _spatial_scale); 
        virtual void bind(std::string _X_i, std::string _rois_i, std::string _batch_indices_i, std::string _Y_o); 
        virtual void build();

        ~RoiAlign() {}
    };
   
}
#endif

