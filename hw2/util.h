#ifndef UTIL_H
#define UTIL_H
#include <mutex>
template <typename T, typename Counter = unsigned char> class SetOnce {
  public:
    SetOnce(const T &initval = T(), const Counter &initcount = 1)
        : val(initval), counter(initcount) {}
    SetOnce(const SetOnce &) = default;
    SetOnce<T, Counter> &operator=(const T &newval) {
        if (counter) {
            --counter;
            val = newval;
            return *this;
        } else
            throw "Some error";
    }
    operator const T &() const { return val; } // "getter"
    T get() { return val; }

  protected:
    T val;
    Counter counter;
};
#endif