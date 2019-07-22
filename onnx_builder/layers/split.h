#include <vector>
namespace backend {
    class Split {
    public:
        Split ();
        ~Split();
    private:

		int axis;
		std::vector<int> split;
    };
}
