#ifndef BATCHNORMALIZATION_H
#define BATCHNORMALIZATION_H //BatchNormalization

//INPUTS:                   X, scale, B, mean, var
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y
//OPTIONAL_OUTPUTS:         mean, var, saved_mean, saved_var
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
            Shape_t X; Shape_t scale; Shape_t B; Shape_t mean; Shape_t var;
            
            //output
            Shape_t Y;
            Shape_t mean; Shape_t var; Shape_t saved_mean; Shape_t saved_var;
        };

        vuh::Program<Specs, Params>* program;

    public:
        BatchNormalization(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        float epsilon; float momentum;
		
        //input
        std::string X; std::string scale; std::string B; std::string mean; std::string var;
        
        //output
        std::string Y;
        std::string mean; std::string var; std::string saved_mean; std::string saved_var;
        //std::vector<uint32_t> output_shape();
   
        ~BatchNormalization(){}
    };
}


namespace backend {    
    BatchNormalization::BatchNormalization(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/batchnormalization.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* BatchNormalization::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
