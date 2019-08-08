#ifndef MIN_H
#define MIN_H //Min

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   min
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class Min : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            
			
            //input
            
            
            //output
            Shape_t min;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Min(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        
		
        //input
        
        
        //output
        std::string min;
        
        //std::vector<uint32_t> output_shape();
   
        ~Min(){}
    };
}


namespace backend {    
    Min::Min(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/min.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* Min::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
