#include <vector>
namespace backend {
    class ReduceL2 {
    public:
        ReduceL2 ();
        ~ReduceL2();
    private:

		std::vector<int> axes;
		int keepdims;
    };
}
