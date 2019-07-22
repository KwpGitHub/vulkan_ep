#include <vector>
namespace backend {
    class ReduceSum {
    public:
        ReduceSum ();
        ~ReduceSum();
    private:

		std::vector<int> axes;
		int keepdims;
    };
}
