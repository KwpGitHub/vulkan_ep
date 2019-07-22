#include <vector>
namespace backend {
    class DictVectorizer {
    public:
        DictVectorizer ();
        ~DictVectorizer();
    private:

		std::vector<int> int64_vocabulary;
		std::vector<string> string_vocabulary;
    };
}
