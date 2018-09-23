#include "CXYNN/CXYNeuronNetwork.h" // add the NN's header
#include "compute-mfcc/Mfcc.h" // 用于解析mfcc
#include <fstream>

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;


void GenMfccFiles() {
	extractMfcc("../data");
}

// 每帧的特征向量的宽度
const int frameWidth = 40;
// 滑动窗口大小
const int windowWidth = 80;

DenseLayer *Input;
ConvLayer *C1; // Conv 1
DropoutLayer *Dp1; // Dropout 1
ConvLayer *S1; // MeanPool 1
ConvLayer *C2; // Conv 2
DropoutLayer *Dp2; // Dropout 2
DenseLayer *D1; // DNN 1
DenseLayer *D2; // DNN 2
DropoutLayer *Dp3; // Dropout 3
DenseLayer *Output;

Estimator_Softmax *estimator;

const bool isTrain = false; // TrainFlag

void buildNetwork() {
	// todo
	Input = new DenseLayer(windowWidth, frameWidth);
	C1 = new ConvLayer(16, 60, 30, 21, 11, 1, 1, 0, 0);
	Dp1 = new DropoutLayer(16, 60, 30, 0.5, isTrain);
	S1 = new ConvLayer(16, 15, 15, 4, 2, 4, 2, 0, 0);
	C2 = new ConvLayer(16, 10, 10, 6, 6, 1, 1, 0, 0);
	Dp2 = new DropoutLayer(16, 10, 10, 0.5, isTrain);
	D1 = new DenseLayer(1, 32);
	D2 = new DenseLayer(1, 128);
	Dp3 = new DropoutLayer(1, 1, 128, 0.5, isTrain);
	Output = new DenseLayer(1, 6); // for 10 case

	estimator = new Estimator_Softmax(Output);

#ifdef ENABLE_CUDA
	Input -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	C1 -> SetActionFunc(kernel_tanh, kernel_tanhDel);
	Dp1 -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	S1 -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	C2 -> SetActionFunc(kernel_tanh, kernel_tanhDel);
	Dp2 -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	D1 -> SetActionFunc(kernel_tanh, kernel_tanhDel);
	D2 -> SetActionFunc(kernel_tanh, kernel_tanhDel);
	Dp3 -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	Output -> SetActionFunc(kernel_Linear, kernel_LinearDel);
#else
	Input -> SetActionFunc(&ActiveFunction::Linear, &ActiveFunction::LinearDel);
	C1 -> SetActionFunc(&ActiveFunction::tanh, &ActiveFunction::tanhDel);
	Dp1 -> SetActionFunc(&ActiveFunction::Linear, &ActiveFunction::LinearDel);
	S1 -> SetActionFunc(&ActiveFunction::Linear, &ActiveFunction::LinearDel);
	C2 -> SetActionFunc(&ActiveFunction::tanh, &ActiveFunction::tanhDel);
	Dp2 -> SetActionFunc(&ActiveFunction::Linear, &ActiveFunction::LinearDel);
	D1 -> SetActionFunc(&ActiveFunction::tanh, &ActiveFunction::tanhDel);
	D2 -> SetActionFunc(&ActiveFunction::tanh, &ActiveFunction::tanhDel);
	Dp3 -> SetActionFunc(&ActiveFunction::Linear, &ActiveFunction::LinearDel);
	Output -> SetActionFunc(&ActiveFunction::Linear, &ActiveFunction::LinearDel);
#endif

	C1 -> InputLayer(Input);
	Dp1 -> InputLayer(C1);
	S1 -> InputLayer(Dp1);
	C2 -> InputLayer(S1);
	Dp2 -> InputLayer(C2);
	D1 -> InputLayer(Dp2);
	D2 -> InputLayer(D1);
	Dp3 -> InputLayer(D2);
	Output -> InputLayer(Dp3);
}

vector <Matrix<double>*> trainData;
vector <Matrix<double>*> trainLabel;

vector <Matrix<double>*> testData;
vector <Matrix<double>*> testLabel;

vector <string> wordList; // 可能出现的单词表
vector <string> dirList;  // 对应的训练数据路径

// 读取单个数据文件
void readSingleData(string dataPathStr, int idx, vector<Matrix<double>*> &data, vector<Matrix<double>*> &label) {
	ifstream fin(dataPathStr);
	int frameNum = 0; fin >> frameNum;
	if(frameNum <= 0) return;
	Matrix<double> tmp(frameNum, frameWidth);
	for(int i = 1; i <= frameNum; i++) {
		for(int j = 1; j <= frameWidth; j++) {
			fin >> tmp(i, j);
		}
	}

	Matrix<double> *lab = new Matrix<double>(1, 1);
	(*lab)(1) = idx;

	// 通过滑动窗口来得到训练数据
	for(int i = 1; i <= frameNum - windowWidth + 1; i++) {
		Matrix<double> *win = new Matrix<double>(windowWidth, frameWidth);
		double len = 0; // 模长
		for(int row = 1; row <= windowWidth; row++)
			for(int col = 1; col <= frameWidth; col++) {
				(*win)(row, col) = tmp(i+row-1, col);
				len += (*win)(row, col) * (*win)(row, col);
			}
		len = sqrt(len);
		if(len == 0) len = 0.00001;
		len /= 100; // 将模长设置为100, 增大刺激的效果
		FOR(x, 1, windowWidth) FOR(y, 1, frameWidth) (*win)(x, y) /= len; // 单位化

		data.push_back(win);
		label.push_back(lab);
	}
	fin.close();
}

// 读取单个文件夹下的训练数据, 并为每个训练数据生产Label
void readCaseData(string dataPathStr, double rate, double idx){
	puts(("Reading: " + dataPathStr).c_str()); // 输出进度
	fs::path dataPath(dataPathStr);
	int count = 0, lim = 500;
	for(auto &p: fs::directory_iterator(dataPath)) ++count;
	if(count > lim) count = lim;
	int trainNum = count * rate; count = 0;
	for(auto &p: fs::directory_iterator(dataPath)) { ++count;
		if(count <= trainNum) readSingleData(p.path().string(), idx, trainData, trainLabel);
		else readSingleData(p.path().string(), idx, testData, testLabel);
		if(count >= lim) break;
	}
}

//读取训练数据， 并按照比例对训练数据和测试数据进行划分
void readTrainData(string dataPathStr, double rate) {
	wordList.clear(); dirList.clear(); // init
	fs::path dataPath(dataPathStr);
	int count = 0, lim = 6;
	for(auto &p: fs::directory_iterator(dataPath)) {
		fs::path curPath = p.path();
		if(fs::is_directory(p.status())) { 
			count++;
			wordList.push_back(curPath.filename().string()); // 将出现的单词添加到单词表
			dirList.push_back(curPath.string() + "/mfcc");   // 将对应的路径添加到路径表
			if (count >= lim) {
				break;
			}
		}
	}
	for (int i = 0; i < wordList.size(); i++) {
		readCaseData(dirList[i], rate, i+1);
	}
}

// 训练
void train() {
	FuncAbstractor functionAbstractor(Input, Output, estimator, 0.1);

	Optimizer optimizer(
			&functionAbstractor,
			0.0001,
			20000,
			trainData,
			trainLabel,
			"train_backup",
			2333,
			-0.20, 0.20, 0.000001,
			100
			);

	// reg the dropout layers
	optimizer.AddDropoutLayer(Dp1);
	optimizer.AddDropoutLayer(Dp2);
	optimizer.AddDropoutLayer(Dp3);

	optimizer.SetSaveStep(5);
	optimizer.TrainFromFile();
	//optimizer.TrainFromNothing();

	optimizer.Save();
}

// 测试
void test() {
	FuncAbstractor functionAbstractor(Input, Output, estimator, 0.1);

	Predictor predictor(
			&functionAbstractor,
			0.05,
			2000,
			trainData,
			trainLabel,
			"train_backup",
			2333,
			-0.20, 0.20, 0.000001,
			100
			);

	// reg the dropout layers
	predictor.AddDropoutLayer(Dp1);
	predictor.AddDropoutLayer(Dp2);
	predictor.AddDropoutLayer(Dp3);

	Matrix<double> *mat;
	FOR(i, 1, wordList.size()) {
		mat = new Matrix<double>(1, 1);
		(*mat)(1) = i;
		predictor.AddCase(i, mat); // 这里的内存泄漏了, 但是现在并不想管
	}

	int correct = 0;
	for (int i = 0; i < testData.size(); i++) {
		int ans = 0;
		//FOR(j, 1, wordList.size()) if ((*(testLabel[i]))(1, j) > 0.5) ans = j;
		ans = (*(testLabel[i]))(1);
		//printf("%d : %d\n", ans-1, predictor.Classify(testData[i]));
		if (ans == predictor.Classify(testData[i])) correct += 1;
		cerr << i << "/" << correct << endl;
		cerr << ans << " " << predictor.Classify(testData[i]) << endl;
	}
	printf("%d/%d\n", correct, testData.size());

}

int main() {
	//GenMfccFiles();
#ifdef ENABLE_CUDA
	cuda_init();
#endif
	buildNetwork();
	readTrainData("../data", 0.8);
	//train();
	test();
	return 0;
}
