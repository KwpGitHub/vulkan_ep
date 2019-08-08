#ifndef NORMALIZER_H
#define NORMALIZER_H //Normalizer

//INPUTS:                   X
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      norm
//OPTIONAL_PARAMETERS_TYPE: int



namespace backend {
    class Normalizer : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int norm;
			
            //input
            Shape_t X;
            
            //output
            Shape_t Y;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Normalizer(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int norm;
		
        //input
        std::string X;
        
        //output
        std::string Y;
        
        //std::vector<uint32_t> output_shape();
   
        ~Normalizer(){}
    };
}


namespace backend {    
    Normalizer::Normalizer(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/normalizer.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* Normalizer::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
