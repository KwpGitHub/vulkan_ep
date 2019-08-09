#ifndef BATCHNORMALIZATION_H
#define BATCHNORMALIZATION_H //BatchNormalization

//INPUTS:                   X_input, scale_input, B_input, mean_input, var_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_input_o
//OPTIONAL_OUTPUTS:         mean_output_o, var_output_o, saved_mean_output_o, saved_var_output_o
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      epsilon, momentum
//OPTIONAL_PARAMETERS_TYPE: float, float



namespace backend {
    class BatchNormalization : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            float epsilon; float momentum;
			
            //input
            Shape_t X_input; Shape_t scale_input; Shape_t B_input; Shape_t mean_input; Shape_t var_input;
            
            //output
            Shape_t Y_input_o;
            Shape_t mean_output_o; Shape_t var_output_o; Shape_t saved_mean_output_o; Shape_t saved_var_output_o;
        };

        vuh::Program<Specs, Params>* program;

    public:
        BatchNormalization(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        float epsilon; float momentum;
		
        //input
        std::string X_input; std::string scale_input; std::string B_input; std::string mean_input; std::string var_input;
        
        //output
        std::string Y_input_o;
        std::string mean_output_o; std::string var_output_o; std::string saved_mean_output_o; std::string saved_var_output_o;
        //std::vector<uint32_t> output_shape();
   
        ~BatchNormalization(){}
    };
}


namespace backend {    
    BatchNormalization::BatchNormalization(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/batchnormalization.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({epsilon, momentum}, 
                            tensor_dict[X_input], tensor_dict[scale_input], tensor_dict[B_input], tensor_dict[mean_input], tensor_dict[var_input],
                            tensor_dict[Y_input_o], tensor_dict[mean_output_o], tensor_dict[var_output_o], tensor_dict[saved_mean_output_o], tensor_dict[saved_var_output_o] );
    }

    vuh::Device* BatchNormalization::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
