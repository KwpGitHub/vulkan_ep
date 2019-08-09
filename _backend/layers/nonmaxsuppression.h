#ifndef NONMAXSUPPRESSION_H
#define NONMAXSUPPRESSION_H //NonMaxSuppression

//INPUTS:                   boxes_input, scores_input
//OPTIONAL_INPUTS:          max_output_boxes_per_class_output, iou_threshold_output, score_threshold_output
//OUTPUS:                   selected_indices_input_o
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
            Shape_t max_output_boxes_per_class_output; Shape_t iou_threshold_output; Shape_t score_threshold_output;
            //output
            Shape_t selected_indices_input_o;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        NonMaxSuppression(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int center_point_box;
		
        //input
        std::string boxes_input; std::string scores_input;
        std::string max_output_boxes_per_class_output; std::string iou_threshold_output; std::string score_threshold_output;
        //output
        std::string selected_indices_input_o;
        
        //std::vector<uint32_t> output_shape();
   
        ~NonMaxSuppression(){}
    };
}


namespace backend {    
    NonMaxSuppression::NonMaxSuppression(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/nonmaxsuppression.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({center_point_box}, 
                            tensor_dict[boxes_input], tensor_dict[scores_input], tensor_dict[max_output_boxes_per_class_output], tensor_dict[iou_threshold_output], tensor_dict[score_threshold_output],
                            tensor_dict[selected_indices_input_o] );
    }

    vuh::Device* NonMaxSuppression::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
