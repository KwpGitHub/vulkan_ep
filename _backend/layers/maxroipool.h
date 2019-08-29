#ifndef MAXROIPOOL_H
#define MAXROIPOOL_H 

#include "../layer.h"

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
//PARAMETER_TYPES:          std::vector<int>
//OPTIONAL_PARAMETERS:      spatial_scale
//OPTIONAL_PARAMETERS_TYPE: float


//class stuff
namespace layers {   

    class MaxRoiPool : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<int> pooled_shape; float spatial_scale;
        std::string X_i; std::string rois_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
       

    public:
        MaxRoiPool(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<int> _pooled_shape,  float _spatial_scale); 
        virtual void bind(std::string _X_i, std::string _rois_i, std::string _Y_o); 
        virtual void build();

        ~MaxRoiPool() {}
    };
   
}
#endif

