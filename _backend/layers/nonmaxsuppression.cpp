#include "NonMaxSuppression.h"
//cpp stuff
namespace backend {    
   
    NonMaxSuppression::NonMaxSuppression() : Layer() { }
       
    vuh::Device* NonMaxSuppression::_get_device() {
        
        return device;
    }
    
    void NonMaxSuppression::init( int _center_point_box) {      
		 center_point_box = _center_point_box; 
  
    }
    
    void NonMaxSuppression::bind(std::string _boxes_input, std::string _scores_input, std::string _max_output_boxes_per_class_input_opt, std::string _iou_threshold_input_opt, std::string _score_threshold_input_opt, std::string _selected_indices_output){
        boxes_input = _boxes_input; scores_input = _scores_input; max_output_boxes_per_class_input_opt = _max_output_boxes_per_class_input_opt; iou_threshold_input_opt = _iou_threshold_input_opt; score_threshold_input_opt = _score_threshold_input_opt; selected_indices_output = _selected_indices_output;
		binding.boxes_input = tensor_dict[boxes_input]->shape();
  		binding.scores_input = tensor_dict[scores_input]->shape();
  		binding.max_output_boxes_per_class_input_opt = tensor_dict[max_output_boxes_per_class_input_opt]->shape();
  		binding.iou_threshold_input_opt = tensor_dict[iou_threshold_input_opt]->shape();
  		binding.score_threshold_input_opt = tensor_dict[score_threshold_input_opt]->shape();
 
		binding.selected_indices_output = tensor_dict[selected_indices_output]->shape();
 
		binding.center_point_box = center_point_box;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/nonmaxsuppression.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[boxes_input]->data(), *tensor_dict[scores_input]->data(), *tensor_dict[max_output_boxes_per_class_input_opt]->data(), *tensor_dict[iou_threshold_input_opt]->data(), *tensor_dict[score_threshold_input_opt]->data(), *tensor_dict[selected_indices_output]->data());
    }



}



