#ifndef PROCESSING_HPP
#define PROCESSING_HPP

#ifdef _WIN32
    #ifdef BUILDING_DLL
        #define EXPORT __declspec(dllexport)
    #else
        #define EXPORT __declspec(dllimport)
    #endif
#else
    #define EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

EXPORT float cmult(int int_param, float float_param);

#ifdef __cplusplus
}
#endif

#endif //PROCESSING_HPP
