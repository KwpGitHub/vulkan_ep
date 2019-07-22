#include <vector>
namespace backend {
    class Conv {
    public:
        Conv ();
        ~Conv();
    private:

		std::string auto_pad;
		std::vector<int> dilations;
		int group;
		std::vector<int> kernel_shape;
		std::vector<int> pads;
		std::vector<int> strides;
    };
}
