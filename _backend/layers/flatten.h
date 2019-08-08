#ifndef FLATTEN_H
#define FLATTEN_H //Flatten

//INPUTS:                   input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int



namespace backend {
    class Flatten : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int axis;
			
            //input
            Shape_t input;
            
            //output
            Shape_t output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Flatten(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int axis;
		
        //input
        std::string input;
        
        //output
        std::string output;
        
        //std::vector<uint32_t> output_shape();
   
        ~Flatten(){}
    };
}


namespace backend {    
    Flatten::Flatten(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/flatten.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* Flatten::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
