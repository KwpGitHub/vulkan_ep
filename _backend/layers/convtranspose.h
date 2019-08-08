#ifndef CONVTRANSPOSE_H
#define CONVTRANSPOSE_H //ConvTranspose

//INPUTS:                   X, W
//OPTIONAL_INPUTS:          B
//OUTPUS:                   Y
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      auto_pad, dilations, group, kernel_shape, output_padding, output_shape, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t, int, Shape_t, Shape_t, Shape_t, Shape_t, Shape_t



namespace backend {
    class ConvTranspose : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t output_padding; Shape_t output_shape; Shape_t pads; Shape_t strides;
			
            //input
            Shape_t X; Shape_t W;
            Shape_t B;
            //output
            Shape_t Y;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        ConvTranspose(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t output_padding; Shape_t output_shape; Shape_t pads; Shape_t strides;
		
        //input
        std::string X; std::string W;
        std::string B;
        //output
        std::string Y;
        
        //std::vector<uint32_t> output_shape();
   
        ~ConvTranspose(){}
    };
}


namespace backend {    
    ConvTranspose::ConvTranspose(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/convtranspose.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* ConvTranspose::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
