#include "NonMaxSuppression.h"

//cpp stuff
namespace backend {    
   
    NonMaxSuppression::NonMaxSuppression(std::string n, int center_point_box) : Layer(n) { }
       
    vuh::Device* NonMaxSuppression::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void NonMaxSuppression::init() {      
    
		binding.boxes_input = tensor_dict[boxes_input]->shape();
  		binding.scores_input = tensor_dict[scores_input]->shape();
  		binding.max_output_boxes_per_class_input_opt = tensor_dict[max_output_boxes_per_class_input_opt]->shape();
  		binding.iou_threshold_input_opt = tensor_dict[iou_threshold_input_opt]->shape();
  		binding.score_threshold_input_opt = tensor_dict[score_threshold_input_opt]->shape();
 
		binding.selected_indices_output = tensor_dict[selected_indices_output]->shape();
 
		binding.center_point_box = center_point_box;
 
    }
    
    void NonMaxSuppression::call(std::string boxes_input, std::string scores_input, std::string max_output_boxes_per_class_input_opt, std::string iou_threshold_input_opt, std::string score_threshold_input_opt, std::string selected_indices_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/nonmaxsuppression.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[boxes_input]->data(), *tensor_dict[scores_input]->data(), *tensor_dict[max_output_boxes_per_class_input_opt]->data(), *tensor_dict[iou_threshold_input_opt]->data(), *tensor_dict[score_threshold_input_opt]->data(), *tensor_dict[selected_indices_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


