#ifndef REVERSESEQUENCE_H
#define REVERSESEQUENCE_H //ReverseSequence

//INPUTS:                   input_input, sequence_lens_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_input_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      batch_axis, time_axis
//OPTIONAL_PARAMETERS_TYPE: int, int



namespace backend {
    class ReverseSequence : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int batch_axis; int time_axis;
			
            //input
            Shape_t input_input; Shape_t sequence_lens_input;
            
            //output
            Shape_t Y_input_o;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        ReverseSequence(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int batch_axis; int time_axis;
		
        //input
        std::string input_input; std::string sequence_lens_input;
        
        //output
        std::string Y_input_o;
        
        //std::vector<uint32_t> output_shape();
   
        ~ReverseSequence(){}
    };
}


namespace backend {    
    ReverseSequence::ReverseSequence(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/reversesequence.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({batch_axis, time_axis}, 
                            tensor_dict[input_input], tensor_dict[sequence_lens_input],
                            tensor_dict[Y_input_o] );
    }

    vuh::Device* ReverseSequence::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
