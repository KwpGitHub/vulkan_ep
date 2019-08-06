#ifndef NONMAXSUPPRESSION_H
#define NONMAXSUPPRESSION_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class NonMaxSuppression : public Layer {
        struct Params{Shape_t boxes_t; Shape_t scores_t; Shape_t max_output_boxes_per_class_t; Shape_t iou_threshold_t; Shape_t score_threshold_t; Shape_t selected_indices_t; int center_point_box_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string boxes; std::string scores; std::string max_output_boxes_per_class; std::string iou_threshold; std::string score_threshold; std::string selected_indices;
        //parameter 
        Shape_t boxes_t; Shape_t scores_t; Shape_t max_output_boxes_per_class_t; Shape_t iou_threshold_t; Shape_t score_threshold_t; Shape_t selected_indices_t; int center_point_box_t;

    public:
        NonMaxSuppression(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            boxes = i[0]; scores = i[1]; max_output_boxes_per_class = i[2]; iou_threshold = i[3]; score_threshold = i[4];
            selected_indices = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/nonmaxsuppression.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({boxes_t, scores_t, max_output_boxes_per_class_t, iou_threshold_t, score_threshold_t, selected_indices_t, center_point_box_t }, tensor_dict[selected_indices], tensor_dict[boxes], tensor_dict[scores], tensor_dict[max_output_boxes_per_class], tensor_dict[iou_threshold], tensor_dict[score_threshold]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["boxes"], boxes_t);
			convert_vec_param(a["scores"], scores_t);
			convert_vec_param(a["max_output_boxes_per_class"], max_output_boxes_per_class_t);
			convert_vec_param(a["iou_threshold"], iou_threshold_t);
			convert_vec_param(a["score_threshold"], score_threshold_t);
			convert_vec_param(a["selected_indices"], selected_indices_t);
			convert_vec_param(a["center_point_box"], center_point_box_t);   
        }

        //Tensor* operator()(const Tensor* t) {            
        //}

		void forward(){
		}

       /* std::vector<uint32_t> output_shape(){
            for(auto t_name : inputs){
                if(tensor_dict.end() == tensor_dict.find(t_name) && layer_dict.end() != layer_dict.find(t_name)){
                    //need to do math
                    return layer_dict[t_name]->output_shape();
                }
                else if (tensor_dict.end() != tensor_dict.find(t_name) && layer_dict.end() == layer_dict.find(t_name)){
                    //need to do math
                    return tensor_dict[t_name]->dims;
                }

            }
            for(auto t_name : outputs){
                if(tensor_dict.end() != tensor_dict.find(t_name) && layer_dict.end() == layer_dict.find(t_name)){
                    return tensor_dict[t_name]->dims;
                }
            }
        }*/

    
        ~NonMaxSuppression(){}

    };
}

#endif
