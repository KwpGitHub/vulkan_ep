#include <vector>
namespace backend {
    class Squeeze {
    public:
        Squeeze ();
        ~Squeeze();
    private:

		std::vector<int> axes;
    };
}
