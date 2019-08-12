#ifndef NONMAXSUPPRESSION_H
#define NONMAXSUPPRESSION_H //NonMaxSuppression

//INPUTS:                   boxes_input, scores_input
//OPTIONAL_INPUTS:          max_output_boxes_per_class_input_o, iou_threshold_input_o, score_threshold_input_o
//OUTPUS:                   selected_indices_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      center_point_box
//OPTIONAL_PARAMETERS_TYPE: int



namespace backend {
    class NonMaxSuppression : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int center_point_box;
			
            //input
            Shape_t boxes_input; Shape_t scores_input;
            Shape_t max_output_boxes_per_class_input_o; Shape_t iou_threshold_input_o; Shape_t score_threshold_input_o;
            //output
            Shape_t selected_indices_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        NonMaxSuppression(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int center_point_box;
		
        //input
        std::string boxes_input; std::string scores_input;
        std::string max_output_boxes_per_class_input_o; std::string iou_threshold_input_o; std::string score_threshold_input_o;
        //output
        std::string selected_indices_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~NonMaxSuppression(){}
    };
}


namespace backend {    
    NonMaxSuppression::NonMaxSuppression(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/nonmaxsuppression.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({center_point_box, tensor_dict[boxes_input]->shape(), tensor_dict[scores_input]->shape(), tensor_dict[max_output_boxes_per_class_input_o]->shape(), tensor_dict[iou_threshold_input_o]->shape(), tensor_dict[score_threshold_input_o]->shape(), tensor_dict[selected_indices_output]->shape()}, 
                            tensor_dict[boxes_input], tensor_dict[scores_input], tensor_dict[max_output_boxes_per_class_input_o], tensor_dict[iou_threshold_input_o], tensor_dict[score_threshold_input_o],
                            tensor_dict[selected_indices_output] );
    }

    vuh::Device* NonMaxSuppression::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
