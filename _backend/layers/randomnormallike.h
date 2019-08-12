#ifndef RANDOMNORMALLIKE_H
#define RANDOMNORMALLIKE_H //RandomNormalLike

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      dtype, mean, scale, seed
//OPTIONAL_PARAMETERS_TYPE: int, float, float, float



namespace backend {
    class RandomNormalLike : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int dtype; float mean; float scale; float seed;
			
            //input
            Shape_t input_input;
            
            //output
            Shape_t output_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        RandomNormalLike(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int dtype; float mean; float scale; float seed;
		
        //input
        std::string input_input;
        
        //output
        std::string output_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~RandomNormalLike(){}
    };
}


namespace backend {    
    RandomNormalLike::RandomNormalLike(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/randomnormallike.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({dtype, mean, scale, seed, tensor_dict[input_input]->shape(), tensor_dict[output_output]->shape()}, 
                            tensor_dict[input_input],
                            tensor_dict[output_output] );
    }

    vuh::Device* RandomNormalLike::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
