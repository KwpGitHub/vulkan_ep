#ifndef RANDOMUNIFORM_H
#define RANDOMUNIFORM_H //RandomUniform

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      dtype, high, low, seed
//OPTIONAL_PARAMETERS_TYPE: int, float, float, float



namespace backend {
    class RandomUniform : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t shape; int dtype; float high; float low; float seed;
			
            //input
            
            
            //output
            Shape_t output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        RandomUniform(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Shape_t shape; int dtype; float high; float low; float seed;
		
        //input
        
        
        //output
        std::string output;
        
        //std::vector<uint32_t> output_shape();
   
        ~RandomUniform(){}
    };
}


namespace backend {    
    RandomUniform::RandomUniform(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/randomuniform.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* RandomUniform::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
