#ifndef NONMAXSUPPRESSION_H
#define NONMAXSUPPRESSION_H 

#include "../layer.h"

/*

Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
result in the same boxes being selected by the algorithm.
The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.
Note: The boxes doesn't has class dimension which means it alwasy has scores calculated for different classes on same box.

input: An input tensor with shape [num_batches, spatial_dimension, 4]. The single box data format is indicated by center_point_box.
input: An input tensor with shape [num_batches, num_classes, spatial_dimension]
input: Integer representing the maximum number of boxes to be selected per batch per class. It is a scalar.
input: Float representing the threshold for deciding whether boxes overlap too much with respect to IOU. It is scalar. Value range [0, 1].
input: Float representing the threshold for deciding when to remove boxes based on score. It is a scalar
output: selected indices from the boxes tensor. [num_selected_indices, 3], the selected index format is [batch_index, class_index, box_index].
*/

//NonMaxSuppression
//INPUTS:                   boxes_i, scores_i
//OPTIONAL_INPUTS:          max_output_boxes_per_class_i, iou_threshold_i, score_threshold_i
//OUTPUS:                   selected_indices_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      center_point_box
//OPTIONAL_PARAMETERS_TYPE: int


//class stuff
namespace layers {   

    class NonMaxSuppression : public backend::Layer {
        typedef struct {          
            backend::Shape_t boxes_i; backend::Shape_t scores_i;
            backend::Shape_t max_output_boxes_per_class_i; backend::Shape_t iou_threshold_i; backend::Shape_t score_threshold_i;
            backend::Shape_t selected_indices_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        int center_point_box;
        std::string boxes_i; std::string scores_i;
        std::string max_output_boxes_per_class_i; std::string iou_threshold_i; std::string score_threshold_i;
        std::string selected_indices_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        NonMaxSuppression(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( int _center_point_box); 
        virtual void bind(std::string _boxes_i, std::string _scores_i, std::string _max_output_boxes_per_class_i, std::string _iou_threshold_i, std::string _score_threshold_i, std::string _selected_indices_o); 
        virtual void build();

        ~NonMaxSuppression() {}
    };
   
}
#endif

