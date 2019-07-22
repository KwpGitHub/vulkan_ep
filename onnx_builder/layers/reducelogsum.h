#include <vector>
namespace backend {
    class ReduceLogSum {
    public:
        ReduceLogSum ();
        ~ReduceLogSum();
    private:

		std::vector<int> axes;
		int keepdims;
    };
}
