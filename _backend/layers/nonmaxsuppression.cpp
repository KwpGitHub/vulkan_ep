#include "nonmaxsuppression.h"
//cpp stuff
namespace layers {    
   
    NonMaxSuppression::NonMaxSuppression(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/nonmaxsuppression.spv");       
        dev = backend::device;
    }
       
        
    void NonMaxSuppression::init( int _center_point_box) {      
		 center_point_box = _center_point_box; 
  

    }
    
    void NonMaxSuppression::bind(std::string _boxes_i, std::string _scores_i, std::string _max_output_boxes_per_class_i, std::string _iou_threshold_i, std::string _score_threshold_i, std::string _selected_indices_o){    
        boxes_i = _boxes_i; scores_i = _scores_i; max_output_boxes_per_class_i = _max_output_boxes_per_class_i; iou_threshold_i = _iou_threshold_i; score_threshold_i = _score_threshold_i; selected_indices_o = _selected_indices_o;        
		SHAPES.push_back(backend::tensor_dict[boxes_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[scores_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[max_output_boxes_per_class_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[iou_threshold_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[score_threshold_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[selected_indices_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void NonMaxSuppression::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(  vuh::div_up(backend::tensor_dict[boxes_i]->shape().w, PROCESSKERNEL_SIZE),
                        vuh::div_up(backend::tensor_dict[boxes_i]->shape().h, PROCESSKERNEL_SIZE), 
                        vuh::div_up(backend::tensor_dict[boxes_i]->shape().d, PROCESSKERNEL_SIZE));
        program->spec(PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE, PROCESSKERNEL_SIZE);
        program->bind({128}, *_SHAPES, *backend::tensor_dict[boxes_i]->data, *backend::tensor_dict[scores_i]->data, *backend::tensor_dict[max_output_boxes_per_class_i]->data, *backend::tensor_dict[iou_threshold_i]->data, *backend::tensor_dict[score_threshold_i]->data, *backend::tensor_dict[selected_indices_o]->data);
    }

    void NonMaxSuppression::forward(){ 
        program->run();
    }

}

