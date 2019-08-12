#ifndef LRN_H
#define LRN_H //LRN

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               size
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      alpha, beta, bias
//OPTIONAL_PARAMETERS_TYPE: float, float, float



namespace backend {
    class LRN : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int size; float alpha; float beta; float bias;
			
            //input
            Shape_t X_input;
            
            //output
            Shape_t Y_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        LRN(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int size; float alpha; float beta; float bias;
		
        //input
        std::string X_input;
        
        //output
        std::string Y_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~LRN(){}
    };
}


namespace backend {    
    LRN::LRN(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/lrn.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({size, alpha, beta, bias, tensor_dict[X_input]->shape(), tensor_dict[Y_output]->shape()}, 
                            tensor_dict[X_input],
                            tensor_dict[Y_output] );
    }

    vuh::Device* LRN::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
