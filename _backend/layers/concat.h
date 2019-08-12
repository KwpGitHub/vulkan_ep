#ifndef CONCAT_H
#define CONCAT_H //Concat

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   concat_result_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               axis
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class Concat : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int axis;
			
            //input
            
            
            //output
            Shape_t concat_result_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Concat(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int axis;
		
        //input
        
        
        //output
        std::string concat_result_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~Concat(){}
    };
}


namespace backend {    
    Concat::Concat(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/concat.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({axis, tensor_dict[concat_result_output]->shape()}, 
                            ,
                            tensor_dict[concat_result_output] );
    }

    vuh::Device* Concat::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
