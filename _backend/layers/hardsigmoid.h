#ifndef HARDSIGMOID_H
#define HARDSIGMOID_H //HardSigmoid

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha, beta
//OPTIONAL_PARAMETERS_TYPE: float, float



namespace backend {
    class HardSigmoid : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            float alpha; float beta;
			
            //input
            Shape_t X_input;
            
            //output
            Shape_t Y_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        HardSigmoid(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        float alpha; float beta;
		
        //input
        std::string X_input;
        
        //output
        std::string Y_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~HardSigmoid(){}
    };
}


namespace backend {    
    HardSigmoid::HardSigmoid(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/hardsigmoid.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({alpha, beta, tensor_dict[X_input]->shape(), tensor_dict[Y_output]->shape()}, 
                            tensor_dict[X_input],
                            tensor_dict[Y_output] );
    }

    vuh::Device* HardSigmoid::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
