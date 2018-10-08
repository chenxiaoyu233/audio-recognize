// -----------------------------------------------------------------------------
// Wrapper for MFCC feature extractor
// -----------------------------------------------------------------------------
//
//  Copyright (C) 2016 D S Pavan Kumar
//  dspavankumar [at] gmail [dot] com
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "mfcc.cc"

// A simple option parser
char* getCmdOption(char **begin, char **end, const std::string &value) {
    char **iter = std::find(begin, end, value);
    if (iter != end && ++iter != end)
        return *iter;
    return nullptr;
}

// Process each file
int processFile (MFCC &mfccComputer, const char* wavPath, const char* mfcPath) {
    // Initialise input and output streams    
    std::ifstream wavFp;
    std::ofstream mfcFp;
    
    // Check if input is readable
    wavFp.open(wavPath);
    if (!wavFp.is_open()) {
        std::cerr << "Unable to open input file: " << wavPath << std::endl;
        return 1;
    }
    
    // Check if output is writable
    mfcFp.open(mfcPath);
    if (!mfcFp.is_open()) {
        std::cerr << "Unable to open output file: " << mfcPath << std::endl;
        wavFp.close();
        return 1;
    }
   
    // Extract and write features
    if (mfccComputer.process (wavFp, mfcFp))
        std::cerr << "Error processing " << wavPath << std::endl;

    wavFp.close();
    mfcFp.close();
    return 0;
}

// Process lists
int processList (MFCC &mfccComputer, const char* wavListPath, const char* mfcListPath) {
    std::ifstream wavListFp, mfcListFp;

    // Check if wav list is readable
    wavListFp.open(wavListPath);
    if (!wavListFp.is_open()) {
        std::cerr << "Unable to open input list: " << wavListPath << std::endl;
        return 1;
    }

    // Check if mfc list is readable
    mfcListFp.open(mfcListPath);
    if (!mfcListFp.is_open()) {
        std::cerr << "Unable to open output list: " << mfcListPath << std::endl;
        return 1;
    }

    // Process lists
    std::string wavPath, mfcPath;
    while (true) {
        std::getline (wavListFp, wavPath);
        std::getline (mfcListFp, mfcPath);
        if (wavPath.empty() || mfcPath.empty()) {
            wavListFp.close();
            mfcListFp.close();
            return 0;
        }
        if (processFile (mfccComputer, wavPath.c_str(), mfcPath.c_str())) {
            wavListFp.close();
            mfcListFp.close();
            return 1;
        }
    }
}

// Main
MFCC* mfccCreate() {
    std::string USAGE = "compute-mfcc : MFCC Extractor\n";
    USAGE += "OPTIONS\n";
    USAGE += "--input           : Input 16 bit PCM Wave file\n";
    USAGE += "--output          : Output MFCC file in CSV format, each frame in a line\n";
    USAGE += "--inputlist       : List of input Wave files\n";
    USAGE += "--outputlist      : List of output MFCC CSV files\n";
    USAGE += "--numcepstra      : Number of output cepstra, excluding log-energy (default=12)\n";
    USAGE += "--numfilters      : Number of Mel warped filters in filterbank (default=40)\n";
    USAGE += "--samplingrate    : Sampling rate in Hertz (default=16000)\n";
    USAGE += "--winlength       : Length of analysis window in milliseconds (default=25)\n";
    USAGE += "--frameshift      : Frame shift in milliseconds (default=10)\n";
    USAGE += "--lowfreq         : Filterbank low frequency cutoff in Hertz (default=50)\n";
    USAGE += "--highfreq        : Filterbank high freqency cutoff in Hertz (default=samplingrate/2)\n";
    USAGE += "USAGE EXAMPLES\n";
    USAGE += "compute-mfcc --input input.wav --output output.mfc\n";
    USAGE += "compute-mfcc --input input.wav --output output.mfc --samplingrate 8000\n";
    USAGE += "compute-mfcc --inputlist input.list --outputlist output.list\n";
    USAGE += "compute-mfcc --inputlist input.list --outputlist output.list --numcepstra 17 --samplingrate 44100\n";

    char *wavListPath = NULL;
    char *mfcListPath = NULL;
    char *numCepstraC = NULL;
    char *numFiltersC = NULL;
    char *samplingRateC = NULL;
    char *winLengthC = NULL;
    char *frameShiftC = NULL;
    char *lowFreqC = NULL;
    char *highFreqC = NULL;

    // Assign variables
    int numCepstra = (numCepstraC ? atoi(numCepstraC) : 40);
    int numFilters = (numFiltersC ? atoi(numFiltersC) : 40);
    int samplingRate = (samplingRateC ? atoi(samplingRateC) : 16000);
    int winLength = (winLengthC ? atoi(winLengthC) : 25);
    int frameShift = (frameShiftC ? atoi(frameShiftC) : 10);
    int lowFreq = (lowFreqC ? atoi(lowFreqC) : 50);
    int highFreq = (highFreqC ? atoi(highFreqC) : samplingRate/2);

    // Initialise MFCC class instance
	return new MFCC (samplingRate, numCepstra, winLength, frameShift, numFilters, lowFreq, highFreq);
}

#include "Mfcc.h"

MFCC *mfccComputer;
int16_t *FrameBuffer;
int len; // 每一帧的输入的长度
deque <double> MfccWindow;

bool MFCC_INIT_FLAG = false; // 保证MFCC只初始化一次

extern "C" void InitMfcc(int N) {
	if(MFCC_INIT_FLAG) return;
	MFCC_INIT_FLAG = true;

	mfccComputer = mfccCreate();
	len = N;
	FrameBuffer = new int16_t[N];
	for(int i = 0; i < frameWidth; i++) 
		for(int j = 0; j < windowWidth; j++) {
			MfccWindow.push_back(0);
		}
}

// 设置当前FrameBuffer的某一位
extern "C" void SetValue(int idx, int16_t val) {
	FrameBuffer[idx] = val;
}

extern "C" void AddFrame() {
	vector<double> spec = mfccComputer -> processFrame(FrameBuffer, len);
	// Debug
	//for( int i = 0; i < spec.size()-1; i++) fprintf(logOut, "%.4lf ", spec[i]);
	//fprintf(logOut, "\n");
	
	// 单位化
	double len = 0; // 模长
	for (int i = 0; i < spec.size()-1; i++) len += spec[i] * spec[i];
	len = sqrt(len);
	for (int i = 0; i < spec.size()-1; i++) spec[i] /= len;
	
	for(int i = 0; i < spec.size()-1; i++) MfccWindow.push_back(spec[i]);
	for(int i = 0; i < spec.size()-1; i++) MfccWindow.pop_front();
}

extern "C" void SetPrev(int idx, int16_t val) {
	mfccComputer -> prevsamples[idx] = val;
}
