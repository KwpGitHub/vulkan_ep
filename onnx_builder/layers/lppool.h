#include <vector>
namespace backend {
    class LpPool {
    public:
        LpPool ();
        ~LpPool();
    private:

		std::string auto_pad;
		std::vector<int> kernel_shape;
		int p;
		std::vector<int> pads;
		std::vector<int> strides;
    };
}
