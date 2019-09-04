#include "nonmaxsuppression.h"
//cpp stuff
namespace layers {    
   
    NonMaxSuppression::NonMaxSuppression(std::string name) : backend::Layer(name) {    
        file.append(backend::file_path);
        file.append("shaders/bin/nonmaxsuppression.spv");       
        dev = backend::g_device;
    }
       
        
    void NonMaxSuppression::init( int _center_point_box) {      
		 m_center_point_box = _center_point_box; 
  

    }
    
    void NonMaxSuppression::bind(std::string _boxes_i, std::string _scores_i, std::string _max_output_boxes_per_class_i, std::string _iou_threshold_i, std::string _score_threshold_i, std::string _selected_indices_o){    
        m_boxes_i = _boxes_i; m_scores_i = _scores_i; m_max_output_boxes_per_class_i = _max_output_boxes_per_class_i; m_iou_threshold_i = _iou_threshold_i; m_score_threshold_i = _score_threshold_i; m_selected_indices_o = _selected_indices_o;        
		SHAPES.push_back(backend::tensor_dict[m_boxes_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_scores_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_max_output_boxes_per_class_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_iou_threshold_i]->shape());
  		SHAPES.push_back(backend::tensor_dict[m_score_threshold_i]->shape());
 
		SHAPES.push_back(backend::tensor_dict[m_selected_indices_o]->shape());
 
        _SHAPES = new vuh::Array<backend::Shape_t>(*dev, SHAPES);


    }

    void NonMaxSuppression::build(){     
        program = new vuh::Program<Specs, binding_descriptor>(*dev, file.c_str());
        program->grid(vuh::div_up(SHAPES[0].w, PROCESSKERNEL_SIZE_x), vuh::div_up(SHAPES[0].h, PROCESSKERNEL_SIZE_y), vuh::div_up(SHAPES[0].d, PROCESSKERNEL_SIZE_z));
        program->spec(PROCESSKERNEL_SIZE_x, PROCESSKERNEL_SIZE_y, PROCESSKERNEL_SIZE_z);
       
    }

    void NonMaxSuppression::forward(){ 
        program->operator()({2, 1}, *_SHAPES, *backend::tensor_dict[m_boxes_i]->data, *backend::tensor_dict[m_scores_i]->data, *backend::tensor_dict[m_max_output_boxes_per_class_i]->data, *backend::tensor_dict[m_iou_threshold_i]->data, *backend::tensor_dict[m_score_threshold_i]->data, *backend::tensor_dict[m_selected_indices_o]->data);
        //program->run();
    }

}

