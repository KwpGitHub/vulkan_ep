#include <vector>
namespace backend {
    class Pad {
    public:
        Pad ();
        ~Pad();
    private:

		std::string mode;
		std::vector<int> pads;
		float value;
    };
}
