#ifndef SLICE_H
#define SLICE_H //Slice

//INPUTS:                   data_input, starts_input, ends_input
//OPTIONAL_INPUTS:          axes_output, steps_output
//OUTPUS:                   output_input_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class Slice : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            
			
            //input
            Shape_t data_input; Shape_t starts_input; Shape_t ends_input;
            Shape_t axes_output; Shape_t steps_output;
            //output
            Shape_t output_input_o;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Slice(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        
		
        //input
        std::string data_input; std::string starts_input; std::string ends_input;
        std::string axes_output; std::string steps_output;
        //output
        std::string output_input_o;
        
        //std::vector<uint32_t> output_shape();
   
        ~Slice(){}
    };
}


namespace backend {    
    Slice::Slice(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/slice.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({}, 
                            tensor_dict[data_input], tensor_dict[starts_input], tensor_dict[ends_input], tensor_dict[axes_output], tensor_dict[steps_output],
                            tensor_dict[output_input_o] );
    }

    vuh::Device* Slice::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
