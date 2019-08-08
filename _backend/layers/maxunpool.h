#ifndef MAXUNPOOL_H
#define MAXUNPOOL_H //MaxUnpool

//INPUTS:                   X, I
//OPTIONAL_INPUTS:          output_shape
//OUTPUS:                   output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      pads, strides
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Shape_t



namespace backend {
    class MaxUnpool : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t kernel_shape; Shape_t pads; Shape_t strides;
			
            //input
            Shape_t X; Shape_t I;
            Shape_t output_shape;
            //output
            Shape_t output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        MaxUnpool(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Shape_t kernel_shape; Shape_t pads; Shape_t strides;
		
        //input
        std::string X; std::string I;
        std::string output_shape;
        //output
        std::string output;
        
        //std::vector<uint32_t> output_shape();
   
        ~MaxUnpool(){}
    };
}


namespace backend {    
    MaxUnpool::MaxUnpool(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/maxunpool.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* MaxUnpool::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
