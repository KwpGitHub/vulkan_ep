#ifndef CONSTANT_H
#define CONSTANT_H //Constant

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               value
//PARAMETER_TYPES:          Tensor*
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class Constant : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            
			Shape_t value;
            //input
            
            
            //output
            Shape_t output_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Constant(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Tensor* value;
		Shape_t value_t;
        //input
        
        
        //output
        std::string output_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~Constant(){}
    };
}


namespace backend {    
    Constant::Constant(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/constant.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({value_t, tensor_dict[output_output]->shape()}, 
                            tensor_dict[value],
                            tensor_dict[output_output] );
    }

    vuh::Device* Constant::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
