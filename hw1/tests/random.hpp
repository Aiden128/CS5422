#include <random>
#include <exception>
#include <vector>
#include <climits>
#include <type_traits>

// Function template: random_gen
// Description: Generate random numbers (int or double)
// Param: (int) Number of samples
// Return: Single vector
template <typename T> const std::vector<T> random_gen(const int num) {
    std::random_device rd;
    std::mt19937_64 generator( rd() );
    std::normal_distribution<T> norm(0.0, 1.0);
    std::uniform_int_distribution<int> uni_int(0, 10000); 
    std::vector<T> vector;

    if (num < 0) throw "Invalid argument";
    for(int i = 0; i < num; ++i){
        if(std::is_same<T, int>::value) {
            vector.push_back(uni_int(generator));
        }
        else {
            vector.push_back(static_cast<T>(norm(generator)));
        }
    }
    return vector;
}
