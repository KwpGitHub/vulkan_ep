#include <vector>
namespace backend {
    class ConvTranspose {
    public:
        ConvTranspose ();
        ~ConvTranspose();
    private:

		std::string auto_pad;
		std::vector<int> dilations;
		int group;
		std::vector<int> kernel_shape;
		std::vector<int> output_padding;
		std::vector<int> output_shape;
		std::vector<int> pads;
		std::vector<int> strides;
    };
}
