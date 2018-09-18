#ifndef __MFCC_H__
#define __MFCC_H__

#include <experimental/filesystem>
#include <string>

namespace fs = std::experimental::filesystem;
using namespace std;
// 使用compute-mfcc从dataPath中计算出mfcc, 并保存为%.mfcc
void extractMfcc(string dataPath);
void computeMfccUnderDir(fs::path subPath);

#endif
