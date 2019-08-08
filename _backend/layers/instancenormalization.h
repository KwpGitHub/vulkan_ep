#ifndef INSTANCENORMALIZATION_H
#define INSTANCENORMALIZATION_H //InstanceNormalization

//INPUTS:                   input, scale, B
//OPTIONAL_INPUTS:          
//OUTPUS:                   output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      epsilon
//OPTIONAL_PARAMETERS_TYPE: float



namespace backend {
    class InstanceNormalization : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            float epsilon;
			
            //input
            Shape_t input; Shape_t scale; Shape_t B;
            
            //output
            Shape_t output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        InstanceNormalization(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        float epsilon;
		
        //input
        std::string input; std::string scale; std::string B;
        
        //output
        std::string output;
        
        //std::vector<uint32_t> output_shape();
   
        ~InstanceNormalization(){}
    };
}


namespace backend {    
    InstanceNormalization::InstanceNormalization(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/instancenormalization.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* InstanceNormalization::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
