#include <vector>
namespace backend {
    class Scaler {
    public:
        Scaler ();
        ~Scaler();
    private:

		std::vector<float> offset;
		std::vector<float> scale;
    };
}
