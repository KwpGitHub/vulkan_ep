#ifndef ARRAYFEATUREEXTRACTOR_H
#define ARRAYFEATUREEXTRACTOR_H //ArrayFeatureExtractor

//INPUTS:                   X, Y
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class ArrayFeatureExtractor : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            
			
            //input
            Shape_t X; Shape_t Y;
            
            //output
            Shape_t Z;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        ArrayFeatureExtractor(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        
		
        //input
        std::string X; std::string Y;
        
        //output
        std::string Z;
        
        //std::vector<uint32_t> output_shape();
   
        ~ArrayFeatureExtractor(){}
    };
}


namespace backend {    
    ArrayFeatureExtractor::ArrayFeatureExtractor(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/arrayfeatureextractor.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* ArrayFeatureExtractor::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
