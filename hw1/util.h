#ifndef UTIL_H
#define UTIL_H
#include <utility>
//class OE_sort{
//public:
//    
//private:
//}

// Note that this implementation is not thread-safe
template <typename T> inline void swap(T &a, T &b) {
    T temp = std::move(a);
    a = std::move(b);
    b = std::move(temp);
}

#endif
