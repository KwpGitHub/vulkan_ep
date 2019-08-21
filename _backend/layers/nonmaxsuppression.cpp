#include "NonMaxSuppression.h"
//cpp stuff
namespace backend {    
   
    NonMaxSuppression::NonMaxSuppression(std::string name) : Layer(name) { }
       
    vuh::Device* NonMaxSuppression::_get_device() {
        
        return device;
    }
    
    void NonMaxSuppression::init( int _center_point_box) {      
		 center_point_box = _center_point_box; 
  
    }
    
    void NonMaxSuppression::bind(std::string _boxes_i, std::string _scores_i, std::string _max_output_boxes_per_class_i, std::string _iou_threshold_i, std::string _score_threshold_i, std::string _selected_indices_o){
        boxes_i = _boxes_i; scores_i = _scores_i; max_output_boxes_per_class_i = _max_output_boxes_per_class_i; iou_threshold_i = _iou_threshold_i; score_threshold_i = _score_threshold_i; selected_indices_o = _selected_indices_o;

		binding.boxes_i = tensor_dict[boxes_i]->shape();
  		binding.scores_i = tensor_dict[scores_i]->shape();
  		binding.max_output_boxes_per_class_i = tensor_dict[max_output_boxes_per_class_i]->shape();
  		binding.iou_threshold_i = tensor_dict[iou_threshold_i]->shape();
  		binding.score_threshold_i = tensor_dict[score_threshold_i]->shape();
 
		binding.selected_indices_o = tensor_dict[selected_indices_o]->shape();
 
		binding.center_point_box = center_point_box;
 

        
    }
}

