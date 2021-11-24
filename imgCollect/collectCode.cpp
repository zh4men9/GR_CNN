// Author: Little-Chen
// Emial: Chenxiuyan_t@163.com

#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <direct.h>
#include <io.h>
#include <time.h>
using namespace std;
using namespace cv;

constexpr int MAXIMG = 5000; //最大采集图片数量
constexpr int IMGSIZE =  128; //图片大小
constexpr double TEST_RATIO = 0.02; //测试集比例

void createDir() { // 创建保存数据集的文件夹
	string dirName = "./data";

	if (0 != _access(dirName.c_str(), 0)) {
		_mkdir(dirName.c_str());
	}

	dirName = "./data/gestures";
	if (0 != _access(dirName.c_str(), 0)) {
		_mkdir(dirName.c_str());
	}

	dirName = "./data/gestures/train";
	if (0 != _access(dirName.c_str(), 0)) {
		_mkdir(dirName.c_str());
	}

	dirName = "./data/gestures/test";
	if (0 != _access(dirName.c_str(), 0)) {
		_mkdir(dirName.c_str());
	}

	dirName = "./data/gestures/train/rock";
	if (0 != _access(dirName.c_str(), 0)) {
		_mkdir(dirName.c_str());
	}

	dirName = "./data/gestures/train/paper";
	if (0 != _access(dirName.c_str(), 0)) {
		_mkdir(dirName.c_str());
	}

	dirName = "./data/gestures/train/scissors";
	if (0 != _access(dirName.c_str(), 0)) {
		_mkdir(dirName.c_str());
	}

	dirName = "./data/gestures/train/others";
	if (0 != _access(dirName.c_str(), 0)) {
		_mkdir(dirName.c_str());
	}

	dirName = "./data/gestures/test/rock";
	if (0 != _access(dirName.c_str(), 0)) {
		_mkdir(dirName.c_str());
	}

	dirName = "./data/gestures/test/paper";
	if (0 != _access(dirName.c_str(), 0)) {
		_mkdir(dirName.c_str());
	}

	dirName = "./data/gestures/test/scissors";
	if (0 != _access(dirName.c_str(), 0)) {
		_mkdir(dirName.c_str());
	}

	dirName = "./data/gestures/test/others";
	if (0 != _access(dirName.c_str(), 0)) {
		_mkdir(dirName.c_str());
	}
}

int main() {

	createDir();
	clock_t start, end;

	VideoCapture capture(0);

	if (capture.isOpened()) {
		cout << "Open camera success!!" << endl << endl;
	}
	else {
		cout << "Open camera failed!!" << endl << endl;
		return 0;
	}

	int gestureType = -1;
	int ctrl = 0;
	stringstream ss;
	string tmp_s = "";
	string savePath = "";

	cout << "请选择采取图像类型(0-石头 1-剪刀 2-布 3-其他 -1-结束程序)：";
	cin >> ctrl;
	while (ctrl != -1) {
		gestureType = ctrl;
		
		Mat frame;
		int cnt = 0;
		char q = ' ';
		int waitTime = 0; //等待10s后开始录制

		do {
			cout << "图像将会在" << 10 - waitTime << "秒后开始录制，可以摁下e或者E直接录制" << endl;

			capture >> frame;
			imshow("Img", frame);

			waitTime++;
			q = waitKey(1000); 
		} while (q != 'e' && q != 'E' && waitTime != 10);//等待10s后开始录制 或者按下e或者E开始录制

		cout << "图像开始采集！" << endl;

		switch (gestureType)
		{

		case 0:

			start = clock();

			while (cnt != MAXIMG + 1) {

				capture >> frame;
				imshow("Img", frame);
				waitKey(1);
				while (cnt < TEST_RATIO * MAXIMG) { //先采集测试集

					capture >> frame;
					imshow("Img", frame);
					waitKey(1);

					ss << cnt;
					ss >> tmp_s;
					ss.clear();
					savePath = "./data/gestures/test/rock/" + tmp_s + ".jpg";

					cnt++;

					resize(frame, frame, Size(IMGSIZE, IMGSIZE));
					imwrite(savePath, frame);
				}

				ss << cnt;
				ss >> tmp_s;
				ss.clear();
				savePath = "./data/gestures/train/rock/" + tmp_s + ".jpg";
				
				resize(frame, frame, Size(IMGSIZE, IMGSIZE));
				imwrite(savePath, frame);

				cnt++;
			}

			end = clock();

			cout << "rock done!!!" << endl;
			cout << "耗时" << double(end - start) / CLOCKS_PER_SEC << "秒！！" << endl;

			break;
		case 1:

			start = clock();

			while (cnt != MAXIMG + 1) {

				capture >> frame;
				imshow("Img", frame);
				waitKey(1);

				while (cnt <= TEST_RATIO * MAXIMG) { //先采集测试集

					capture >> frame;
					imshow("Img", frame);
					waitKey(1);
					ss << cnt;
					ss >> tmp_s;
					ss.clear();
					savePath = "./data/gestures/test/scissors/" + tmp_s + ".jpg";

					cnt++;

					resize(frame, frame, Size(IMGSIZE, IMGSIZE));
					imwrite(savePath, frame);
				}

				ss << cnt;
				ss >> tmp_s;
				ss.clear();
				savePath = "./data/gestures/train/scissors/" + tmp_s + ".jpg";

				resize(frame, frame, Size(IMGSIZE, IMGSIZE));
				imwrite(savePath, frame);

				cnt++;
			}

			end = clock();

			cout << "scissors done!!!" << endl;
			cout << "耗时" << double(end - start) / CLOCKS_PER_SEC << "秒！！" << endl;

			break;
		case 2:

			start = clock();

			while (cnt != MAXIMG + 1) {

				capture >> frame;
				imshow("Img", frame);
				waitKey(1);

				while (cnt <= TEST_RATIO * MAXIMG) { //先采集测试集

					capture >> frame;
					imshow("Img", frame);
					waitKey(1);
					ss << cnt;
					ss >> tmp_s;
					ss.clear();
					savePath = "./data/gestures/test/paper/" + tmp_s + ".jpg";

					cnt++;

					resize(frame, frame, Size(IMGSIZE, IMGSIZE));
					imwrite(savePath, frame);
				}

				ss << cnt;
				ss >> tmp_s;
				ss.clear();
				savePath = "./data/gestures/train/paper/" + tmp_s + ".jpg";
				
				resize(frame, frame, Size(IMGSIZE, IMGSIZE));
				imwrite(savePath, frame);

				cnt++;
			}

			end = clock();

			cout << "paper done!!!" << endl;
			cout << "耗时" << double(end - start) / CLOCKS_PER_SEC << "秒！！" << endl;

			break;

		case 3:

			start = clock();

			while (cnt != MAXIMG + 1) {

				capture >> frame;
				imshow("Img", frame);
				waitKey(1);

				while (cnt <= TEST_RATIO * MAXIMG) { //先采集测试集

					capture >> frame;
					imshow("Img", frame);
					waitKey(1);
					ss << cnt;
					ss >> tmp_s;
					ss.clear();
					savePath = "./data/gestures/test/others/" + tmp_s + ".jpg";

					cnt++;

					resize(frame, frame, Size(IMGSIZE, IMGSIZE));
					imwrite(savePath, frame);
				}

				ss << cnt;
				ss >> tmp_s;
				ss.clear();
				savePath = "./data/gestures/train/others/" + tmp_s + ".jpg";
				
				resize(frame, frame, Size(IMGSIZE, IMGSIZE));
				imwrite(savePath, frame);

				cnt++;
			}

			end = clock();

			cout << "others done!!!" << endl;
			cout << "耗时" << double(end - start) / CLOCKS_PER_SEC << "秒！！" << endl;

			break;
		default:
			cout << "input error!!!" << endl;
			break;
		}

		cout << "请选择采取图像类型(0-石头 1-剪刀 2-布 -1-结束程序)：";
		cin >> ctrl;
	}


	return 0;
}