#include "CXYNN/CXYNeuronNetwork.h" // add the NN's header
#include <fstream>

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

const int isTrain = 1; // TrainFlag

void buildNetwork() {
	// todo
	Input = new DenseLayer(windowWidth, frameWidth);
	C1 = new ConvLayer(16, 60, 30, 21, 11, 1, 1, 0, 0);
	//Dp1 = new DropoutLayer(16, 11, 33, 0.8, isTrain);
	S1 = new ConvLayer(16, 15, 15, 4, 2, 4, 2, 0, 0);
	C2 = new ConvLayer(16, 10, 10, 6, 6, 1, 1, 0, 0);
	//Dp2 = new DropoutLayer(16, 2, 8, 0.8, isTrain);
	D1 = new DenseLayer(1, 32);
	D2 = new DenseLayer(1, 128);
	//Dp3 = new DropoutLayer(1, 1, 128, 0.8, isTrain);
	Output = new DenseLayer(1, 6); // for 10 case

	estimator = new Estimator_Softmax(Output);

#ifdef ENABLE_CUDA
	Input -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	C1 -> SetActionFunc(kernel_tanh, kernel_tanhDel);
	//Dp1 -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	S1 -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	C2 -> SetActionFunc(kernel_tanh, kernel_tanhDel);
	//Dp2 -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	D1 -> SetActionFunc(kernel_tanh, kernel_tanhDel);
	D2 -> SetActionFunc(kernel_tanh, kernel_tanhDel);
	//Dp3 -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	Output -> SetActionFunc(kernel_Linear, kernel_LinearDel);
#else
	Input -> SetActionFunc(&ActiveFunction::Linear, &ActiveFunction::LinearDel);
	C1 -> SetActionFunc(&ActiveFunction::tanh, &ActiveFunction::tanhDel);
	//Dp1 -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	S1 -> SetActionFunc(&ActiveFunction::Linear, &ActiveFunction::LinearDel);
	C2 -> SetActionFunc(&ActiveFunction::tanh, &ActiveFunction::tanhDel);
	//Dp2 -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	D1 -> SetActionFunc(&ActiveFunction::tanh, &ActiveFunction::tanhDel);
	D2 -> SetActionFunc(&ActiveFunction::tanh, &ActiveFunction::tanhDel);
	//Dp3 -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	Output -> SetActionFunc(&ActiveFunction::Linear, &ActiveFunction::LinearDel);
#endif

	C1 -> InputLayer(Input);
	//Dp1 -> InputLayer(C1);
	S1 -> InputLayer(C1);
	C2 -> InputLayer(S1);
	//Dp2 -> InputLayer(C2);
	D1 -> InputLayer(C2);
	D2 -> InputLayer(D1);
	//Dp3 -> InputLayer(D2);
	Output -> InputLayer(D2);
}

vector <Matrix<double>*> trainData;
vector <Matrix<double>*> trainLabel;

vector <Matrix<double>*> testData;
vector <Matrix<double>*> testLabel;

vector <string> wordList; // 可能出现的单词表
vector <string> dirList;  // 对应的训练数据路径

// 测试
void test() {
	FuncAbstractor functionAbstractor(Input, Output, estimator, 0.1);

	Predictor predictor(
		&functionAbstractor,
		0.1,
		2000,
		trainData,
		trainLabel,
		"train_backup",
		2333,
		-0.20, 0.20, 0.000001,
		100
	);

	// reg the dropout layers
	//predictor.AddDropoutLayer(Dp1);
	//predictor.AddDropoutLayer(Dp2);
	//predictor.AddDropoutLayer(Dp3);

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
	test();
	return 0;
}
