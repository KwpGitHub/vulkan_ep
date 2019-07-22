#include <vector>
namespace backend {
    class ReduceL1 {
    public:
        ReduceL1 ();
        ~ReduceL1();
    private:

		std::vector<int> axes;
		int keepdims;
    };
}
