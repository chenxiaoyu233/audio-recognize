#ifndef __MFCC_H__
#define __MFCC_H__

#include <string>
#include <deque>
using namespace std;

extern FILE *logOut;
extern "C" __declspec(dllexport) void InitMfcc(int len);
extern "C" __declspec(dllexport) void SetValue(int idx, int16_t val);
extern "C" __declspec(dllexport) void AddFrame();
extern "C" __declspec(dllexport) void SetPrev(int idx, int16_t val);

extern deque<double> MfccWindow;
// 每帧的特征向量的宽度
const int frameWidth = 40;
// 滑动窗口大小
const int windowWidth = 30;

#endif
