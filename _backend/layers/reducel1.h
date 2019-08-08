#ifndef REDUCEL1_H
#define REDUCEL1_H //ReduceL1

//INPUTS:                   data
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes, keepdims
//OPTIONAL_PARAMETERS_TYPE: Shape_t, int



namespace backend {
    class ReduceL1 : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t axes; int keepdims;
			
            //input
            Shape_t data;
            
            //output
            Shape_t reduced;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        ReduceL1(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Shape_t axes; int keepdims;
		
        //input
        std::string data;
        
        //output
        std::string reduced;
        
        //std::vector<uint32_t> output_shape();
   
        ~ReduceL1(){}
    };
}


namespace backend {    
    ReduceL1::ReduceL1(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/reducel1.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* ReduceL1::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
