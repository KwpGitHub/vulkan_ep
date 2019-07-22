#include <vector>
namespace backend {
    class ReduceSumSquare {
    public:
        ReduceSumSquare ();
        ~ReduceSumSquare();
    private:

		std::vector<int> axes;
		int keepdims;
    };
}
