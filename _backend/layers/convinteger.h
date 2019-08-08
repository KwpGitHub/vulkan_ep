#ifndef CONVINTEGER_H
#define CONVINTEGER_H //ConvInteger

//INPUTS:                   x, w
//OPTIONAL_INPUTS:          x_zero_point, w_zero_point
//OUTPUS:                   y
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      auto_pad, dilations, group, kernel_shape, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t, int, Shape_t, Shape_t, Shape_t



namespace backend {
    class ConvInteger : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
			
            //input
            Shape_t x; Shape_t w;
            Shape_t x_zero_point; Shape_t w_zero_point;
            //output
            Shape_t y;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        ConvInteger(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
		
        //input
        std::string x; std::string w;
        std::string x_zero_point; std::string w_zero_point;
        //output
        std::string y;
        
        //std::vector<uint32_t> output_shape();
   
        ~ConvInteger(){}
    };
}


namespace backend {    
    ConvInteger::ConvInteger(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/convinteger.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* ConvInteger::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
