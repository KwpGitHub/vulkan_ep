#ifndef NONMAXSUPPRESSION_H
#define NONMAXSUPPRESSION_H 
#include <pybind11/pybind11.h>
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
//INPUTS:                   boxes_input, scores_input
//OPTIONAL_INPUTS:          max_output_boxes_per_class_input_opt, iou_threshold_input_opt, score_threshold_input_opt
//OUTPUS:                   selected_indices_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      center_point_box
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//class stuff
namespace backend {   

    class NonMaxSuppression : public Layer {
        typedef struct {    
            int center_point_box;
        } parameter_descriptor;  

        typedef struct {
            Tensor* boxes_input; Tensor* scores_input;
            Tensor* max_output_boxes_per_class_input_opt; Tensor* iou_threshold_input_opt; Tensor* score_threshold_input_opt;
        } input_desriptor;

        typedef struct {
            Tensor* selected_indices_output;
            
        } output_descriptor;

        typedef struct {
            int center_point_box;
		
            Shape_t boxes_input; Shape_t scores_input;
            Shape_t max_output_boxes_per_class_input_opt; Shape_t iou_threshold_input_opt; Shape_t score_threshold_input_opt;
            Shape_t selected_indices_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        NonMaxSuppression(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~NonMaxSuppression() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    NonMaxSuppression::NonMaxSuppression(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/nonmaxsuppression.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* NonMaxSuppression::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void NonMaxSuppression::init() {
		binding.boxes_input = input.boxes_input->shape();
  		binding.scores_input = input.scores_input->shape();
  		binding.max_output_boxes_per_class_input_opt = input.max_output_boxes_per_class_input_opt->shape();
  		binding.iou_threshold_input_opt = input.iou_threshold_input_opt->shape();
  		binding.score_threshold_input_opt = input.score_threshold_input_opt->shape();
 
		binding.selected_indices_output = output.selected_indices_output->shape();
 
		binding.center_point_box = parameters.center_point_box;
 
        program->bind(binding, *input.boxes_input->data(), *input.scores_input->data(), *input.max_output_boxes_per_class_input_opt->data(), *input.iou_threshold_input_opt->data(), *input.score_threshold_input_opt->data(), *output.selected_indices_output->data());
    }
    
    void NonMaxSuppression::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<NonMaxSuppression, Layer>(m, "NonMaxSuppression")
            .def("forward", &NonMaxSuppression::forward);    
    }
}*/

#endif
