// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <vector>
#include <queue>

using namespace std;
struct qu {
	int i;
	int j;
}q;
void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}
void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}
void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}
void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}
void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}
void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}
void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}
void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}
void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}
void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}
void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

bool isInside(const Mat& src, int i, int j) {
	return (i >= 0 && i < src.rows&& j >= 0 && j < src.cols);
}
void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}
void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}
void lab4_trasaturi(Mat_<uchar> img) {
	int arie = 0;
	float rb = 0, cb = 0;
	for (int r = 0; r < img.rows; r++)
	{
		for (int c = 0; c < img.cols; c++) {
			if (img(r, c) == 0) {
				arie += 1;
				rb += r;
				cb += c;
			}

		}
	}
	rb /= arie;
	cb /= arie;

	Mat_<uchar> dst = img.clone();
	if (isInside(dst, rb, cb))
		dst.at<uchar>(rb, cb) = 255;
	//desenare cruce in jurul lui rb,cb
	printf("%d \n", arie);
	printf("Centru of masa: %f \n", rb);
	printf("Centru of masa: %f \n", cb);

	imshow("centru de masa", dst);
	float f3 = 0, f2 = 0;
	for (int r = 0; r < img.rows; r++)
	{
		for (int c = 0; c < img.cols; c++) {
			if (img(r, c) == 0) {
				f3 = f3 + (2 * (r - rb) * (c - cb));
				f2 = f2 + pow((c - cb), 2) - pow((r - rb), 2);
			}
		}
	}
	double phi = atan2(f3, f2) / 2;
	printf("Axa de alungire: %f \n", phi);

	float r1 = rb + sin(phi) * 50;
	float c1 = cb + cos(phi) * 50;
	line(dst, Point(cb, rb), Point(c1, r1), 255, 1, 8, 0);
	imshow("imagine", dst);
	int perimetru = 0;
	int rmin = 999999, cmin = 999999, rmax = -1, cmax = -1;
	for (int r = 0; r < img.rows; r++)
	{
		for (int c = 0; c < img.cols; c++) {
			
				if (img(r, c) == 0) {
					if (img(r - 1, c) != 0 || img(r, c - 1) != 0 || img(r, c + 1) != 0 || img(r + 1, c) != 0 || img(r - 1, c - 1) != 0 || img(r + 1, c + 1) != 0 || img(r - 1, c + 1) || img(r + 1, c - 1))
						perimetru++;
					if (rmin > r)
						rmin = r;
					if (rmax < r)
						rmax = r;
					if (cmin > c)
						cmin = c;
					if (cmax < c)
						cmax = c;
				}
			
		}
	}

	printf("Perimetru: %d \n", perimetru);

	float subtiere = 4 * ((float)arie / pow(perimetru, 2)) * PI;
	printf("factprul de subtiere: %f\n", subtiere);

	float ratio = (float)(cmax - cmin + 1) / (float)(rmax - rmin + 1);
	printf("Elongatia,factorul de aspect: %f\n", ratio);

	Mat horz = Mat(img.rows, img.cols, CV_8UC1);
	Mat vert = Mat(img.rows, img.cols, CV_8UC1);
	int hk;
	for (int c = 0; c < img.cols; c++)
	{
		hk = 0;
		for (int r = 0; r < img.rows; r++) {
			if (img(r, c) == 0) {
				hk++;
				horz.at<uchar>(hk, c) = 0;
			}
		}
	}
	imshow("orizontal", horz);
	int vk;
	for (int r = 0; r < img.rows; r++)
	{
		vk = 0;
		for (int c = 0; c < img.cols; c++) {
			if (img(r, c) == 0) {
				vk++;
				vert.at<uchar>(r, vk) = 0;
			}
		}
	}
	imshow("vertical", vert);
}
void onMouse(int event, int x, int y, int flags, void* param) {

	if (event == EVENT_LBUTTONDOWN) {
		cout << x << " " << y << "\n";
		Mat_<Vec3b> img = *(Mat_<Vec3b>*)param;
		Vec3b color = img(y, x);
		cout << color << endl;
		Mat_<uchar> gray(img.rows, img.cols);
		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				gray(i, j) = img(i, j) == color ? 0 : 255;
			}
		}
		lab4_trasaturi(gray);
	}
}
void labelingBFS() {
	uchar val, label = 0;
	int di[8] = { -1,-1,-1, 0, 0, 1,1,1 };
	int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		Mat labels(img.rows, img.cols, CV_32SC1, Scalar(0));

		Mat dst(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));

		for (int i = 1; i < img.rows - 1; i++) {
			for (int j = 1; j < img.cols - 1; j++) {
				val = img.at<uchar>(i, j);
				if ((val == 0) && (labels.at<int>(i, j) == 0)) {
					label++;
					std::queue<Point> Q;
					labels.at<int>(i, j) = label;
					Q.push({ i,j });
					while (!Q.empty()) {
						Point q = Q.front();
						Q.pop();
						for (int k = 0; k < 8; k++) {
							if ((img.at<uchar>(q.x + di[k], q.y + dj[k]) == 0) && (labels.at<int>(q.x + di[k], q.y + dj[k]) == 0)) {
								labels.at<int>(q.x + di[k], q.y + dj[k]) = label;
								Q.push({ q.x + di[k], q.y + dj[k] });
							}
						}
					}
				}
			}
		}

		std::vector<Vec3b> color1 = { Vec3b(255,255,255) };

		srand(time(NULL));

		for (int k = 1; k <= label; k++) {
			color1.push_back(Vec3b(rand() % 256, rand() % 256, rand() % 256));
		}
		for (int i = 1; i < img.rows - 1; i++) {
			for (int j = 1; j < img.cols - 1; j++) {
				if (labels.at<int>(i, j) != 0)
					dst.at<Vec3b>(i, j) = color1[labels.at<int>(i, j)];
			}
		}

		imshow("labels", dst);

		waitKey(0);

	}
}
void testSchimbNivGriFactorAditiv(int additiveFactor)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = src.at<uchar>(i, j);
				int new_val= val + additiveFactor;
				if (new_val < 0)
					new_val = 0;
				else if (new_val > 255)
					new_val = 255;
				uchar gri = new_val;
				dst.at<uchar>(i, j) = gri;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("gri image", dst);
		waitKey();
	}
}
void testSchimbNivGriFactorMultiplicativ(int multiplicativeFactor)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = src.at<uchar>(i, j);
				int new_val = val* multiplicativeFactor;
				if (new_val < 0)
					new_val = 0;
				else if (new_val > 255)
					new_val = 255;
				uchar gri = new_val;
				dst.at<uchar>(i, j) = gri;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);


		imshow("input image", src);
		imshow("gri image", dst);
		bool ok= imwrite("img11.jpg", dst);
		if (ok)
			cout << "Succes";
		else cout << "Eroare";
		waitKey();
	}
}
void testProblema5Lab1()
{
	//127-128
	Mat img(255, 255, CV_8UC3);
	int height = img.rows;
	int width = img.cols;
	
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			//Vec3b v3 = img.at<Vec3b>(i, j);
			//uchar b = v3[0];
			//uchar g = v3[1];
			//uchar r = v3[2];
			
			if (i < 128 && j<128)
			{
				//alb
				img.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else if (i < 128 && j >= 128) {
				//rosu
				img.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
			}
			else if (i >= 128 && j < 128) {
				//verde
				img.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
			}
			else if (i>=128 && j>=128) {
				//galben
				img.at<Vec3b>(i, j) = Vec3b(0, 255, 255);
			}
		}
	imshow("img", img);
	waitKey();
}
void testL2P1() {


	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;

		Mat R = Mat(height, width, CV_8UC3);
		Mat G = Mat(height, width, CV_8UC3);
		Mat B = Mat(height, width, CV_8UC3);

		
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				R.at<Vec3b>(i, j) = Vec3b(0, 0, v3[2]);
				B.at<Vec3b>(i, j) = Vec3b(v3[0], 0, 0);
				G.at<Vec3b>(i, j) = Vec3b(0, v3[1], 0);
			}
		imshow("imaginea", src);
		imshow("R", R);
		imshow("G", G);
		imshow("B", B);
		waitKey();
	}

	
}
void testL2P2() {
	//color to grayscale
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat gray = Mat(height, width, CV_8UC1);
		for(int i=0;i<height;i++)
			for (int j = 0; j < width; j++) {
				Vec3b v3 = src.at<Vec3b>(i, j);
				gray.at<uchar>(i, j) = (v3[0] + v3[1] + v3[2]) / 3;
			}
		imshow("imagine", src);
		imshow("gray", gray);
		waitKey();
	}
}
void testL2P3(int threshold) {
	// grayscale to alb negru
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_8UC1);
		int height = src.rows;
		int width = src.cols;
		Mat gray = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar v = src.at<uchar>(i, j);
				int val = v;
				if (val < threshold)
					gray.at<uchar>(i, j) = 0;
				else
					gray.at<uchar>(i, j) = 255;

			}
		imshow("imagine", src);
		imshow("gray", gray);
		waitKey();
	}
}
void testL2P4() {
	//RBG to HVS
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		Mat Hmat = Mat(height, width, CV_8UC1);
		Mat Vmat = Mat(height, width, CV_8UC1);
		Mat Smat = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar B = v3[0];
				uchar G = v3[1];
				uchar R = v3[2];
				float r = (float) R / 255;
				float g = (float) G / 255;
				float b = (float) B / 255;

				float M = max(max(r, g), b);
				float m= min(min(r, g), b);

				float c = M - m;

				float H, S;
				// stauration
				float V = M;
				if (V != 0)
					S = c / V;
				else S = 0;
				//hue
				if (c != 0)
				{
					if (M == r) H = 60 * (g - b) / c;
					if (M == g) H = 120 + 60*(b - r) / c;
					if (M == b) H = 240 + 60*(r - g) / c;
				}
				else {
					H = 0;
				}
				if (H < 0) H = H + 360;
				Hmat.at<uchar>(i,j) =  H * 255 / 360;
				Smat.at<uchar>(i, j) = S * 255;
				Vmat.at<uchar>(i, j) = V * 255;
			}
		imshow("H", Hmat);
		imshow("V", Vmat);
		imshow("S", Smat);
		waitKey();
	}
}
void testL2P5(int i,int j) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname);
		if (isInside(src, i, j)) {
			imshow("imagine", src);
			waitKey();
		}
		else
			cout << "invalid input";
	}
	
}
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

int* histograma2(const Mat& src) {
	int* histogram = new int[256] {0};

	int height = src.rows;
	int width = src.cols;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int intensity = src.at<uchar>(i, j);
			histogram[intensity]++;
		}
	}

	return histogram;
}
double* compute(const Mat& src) {
	double* pdf = new double[256] {0.0};

	int height = src.rows;
	int width = src.cols;
	int totalPixels = height * width;

	int* histogram = histograma2(src);

	for (int i = 0; i < 256; i++) {
		pdf[i] = static_cast<double>(histogram[i]) / totalPixels;
	}

	delete[] histogram;

	return pdf;
}
void histogramTransformation() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		uchar BLACK = 0;
		uchar WHITE = 255;
		int height = src.rows;
		int width = src.cols;
		Mat bright = Mat(height, width, CV_8UC1, Scalar(WHITE));
		Mat contrast = Mat(height, width, CV_8UC1, Scalar(WHITE));
		Mat gamma = Mat(height, width, CV_8UC1, Scalar(WHITE));
		Mat equalization = Mat(height, width, CV_8UC1, Scalar(WHITE));

		int* histogram = histograma2(src);

		int brightnessOffset = 60;
		double outMin = 10;
		double outMax = 150;
		double gammaValue = 3.0;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int val = src.at<uchar>(i, j);
				if (val + brightnessOffset > 255) {
					val = 255;
				}
				else {
					val += brightnessOffset;
				}
				bright.at<uchar>(i, j) = val;
			}
		}

		double minIn, maxIn;

		for (int i = 0; i < 256; i++) {
			if (histogram[i] > 0) {
				minIn = i;
				break;
			}
		}
		for (int i = 255; i >= 0; i--) {
			if (histogram[i] > 0) {
				maxIn = i;
				break;
			}
		}
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				double in = src.at<uchar>(i, j);
				int out = outMin + (in - minIn) * ((outMax - outMin) / (maxIn - minIn));
				contrast.at<uchar>(i, j) = out;
			}
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				double in = src.at<uchar>(i, j);
				int out = 255 * pow((in / 255.0), gammaValue);
				gamma.at<uchar>(i, j) = out;
			}
		}

		double* pdf = compute(src);
		double cdf[256] = {};
		cdf[0] = pdf[0];
		for (int i = 1; i < 256; i++) {
			cdf[i] = cdf[i - 1] + pdf[i];
		}
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int in = src.at<uchar>(i, j);
				int out = 255 * cdf[in];
				equalization.at<uchar>(i, j) = out;
			}
		}

		imshow("Src", src);
		imshow("Bright", bright);
		imshow("Contrast", contrast);
		imshow("Gamma", gamma);
		imshow("Equalization", equalization);

		waitKey(0);
	}
}
vector<int> calculateIntensityHistogram(const Mat& image) {
	vector<int> histogram(256, 0);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			int intensity = image.at<uchar>(i, j);
			histogram[intensity]++;
		}
	}
	return histogram;
}
void testL3P1() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		vector<int> v = calculateIntensityHistogram(src);

		for (int i = 0; i < 256; i++)
			cout << "i=" << i << " v=" << v[i] << "\n";

		imshow("imaginea", src);
		showHistogram("histogram", v.data(), 256, 500);
		waitKey();
	}
}
void praguriMultipleL3() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		int h[256] = { 0 };
		int g;
		//normalizam histograma
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				g = src.at<uchar>(i, j);
				h[g]++;

			}
		}

		double sum = 0;
		double p[256] = { 0 };
		for (int i = 0; i < 256; i++)
		{
			sum += h[i];
		}

		for (int i = 0; i < 256; i++)
		{
			p[i] = (double)h[i] / sum;
		}

		int WH = 5;
		double TH = 0.0003;
		float v;
		int val;
		int vec[255];
		int count = 1;


		for (int k = WH; k <= 255 - WH; k++) {
			v = 0;
			for (int i = k - WH; i <= k + WH; i++) {
				v += p[i];
				if (p[i] > p[k])
					v += 1000;
			}
			v = v / (2 * WH + 1);
			if (p[k] > v + TH) {
				vec[count] = k;
				count++;
			}
		}
		vec[0] = 0;
		vec[count] = 255;

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				val = src.at<uchar>(i, j);
				int min = 255;
				int close;
				for (int z = 0; z < count + 1; z++) {
					if (abs(val - vec[z]) < min) {
						min = abs(val - vec[z]);
						close = vec[z];
					}
				}
				src.at<uchar>(i, j) = close;
			}
		}


		imshow("input image", src);


		waitKey();
	}

}
void floyd() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		int h[256] = { 0 };
		int g;
		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				g = src.at<uchar>(i, j);
				h[g]++;

			}
		}

		double sum = 0;
		double p[256] = { 0 };
		for (int i = 0; i < 256; i++)
		{
			sum += h[i];
		}

		for (int i = 0; i < 256; i++)
		{
			p[i] = (double)h[i] / sum;
		}

		int WH = 5;
		double TH = 0.0003;
		float v;
		int val;
		int vec[255];
		int count = 1;


		for (int k = WH; k <= 255 - WH; k++) {
			v = 0;
			for (int i = k - WH; i <= k + WH; i++) {
				v += p[i];
				if (p[i] > p[k])
					v += 1000;
			}
			v = v / (2 * WH + 1);
			if (p[k] > v + TH) {
				vec[count] = k;
				count++;
			}
		}
		vec[0] = 0;
		vec[count] = 255;
		int eroare;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				val = src.at<uchar>(i, j);
				int min = 255;
				int close;
				for (int z = 0; z < count + 1; z++) {
					if (abs(val - vec[z]) < min) {
						min = abs(val - vec[z]);
						close = vec[z];
					}
				}
				src.at<uchar>(i, j) = close;

				eroare = val - close;
				if (isInside(src, i + 1, j) == true) {
					if ((src.at<uchar>(i + 1, j) + 7 * eroare / 16) < 0)
						src.at<uchar>(i + 1, j) = 0;
					else if ((src.at<uchar>(i + 1, j) + 7 * eroare / 16) > 255)
						src.at<uchar>(i + 1, j) = 255;
					else
						src.at<uchar>(i + 1, j) = src.at<uchar>(i + 1, j) + 7 * eroare / 16;
				}

				if (isInside(src, i - 1, j + 1) == true) {
					if ((src.at<uchar>(i - 1, j + 1) + 3 * eroare / 16) < 0)
						src.at<uchar>(i - 1, j + 1) = 0;
					else if ((src.at<uchar>(i - 1, j + 1) + 3 * eroare / 16) > 255)
						src.at<uchar>(i - 1, j + 1) = 255;
					else
						src.at<uchar>(i - 1, j + 1) = src.at<uchar>(i - 1, j + 1) + 3 * eroare / 16;
				}

				if (isInside(src, i, j + 1) == true) {
					if ((src.at<uchar>(i, j + 1) + 5 * eroare / 16) < 0)
						src.at<uchar>(i, j + 1) = 0;
					else if ((src.at<uchar>(i, j + 1) + 5 * eroare / 16) > 255)
						src.at<uchar>(i, j + 1) = 255;
					else
						src.at<uchar>(i, j + 1) = src.at<uchar>(i, j + 1) + 5 * eroare / 16;
				}
				if (isInside(src, i + 1, j + 1) == true) {
					if ((src.at<uchar>(i + 1, j + 1) + eroare / 16) < 0)
						src.at<uchar>(i + 1, j + 1) = 0;
					else if ((src.at<uchar>(i + 1, j + 1) + eroare / 16) > 255)
						src.at<uchar>(i + 1, j + 1) = 255;
					else
						src.at<uchar>(i + 1, j + 1) = src.at<uchar>(i + 1, j + 1) + eroare / 16;
				}

			}
		}

		imshow("Floyd Mayweather", src);
		waitKey();
	}

}
vector <float> FDP(vector<int> v, int img_height, int img_width) {
	vector <float> p(256, 0);
	int M = img_height * img_width;
	for (int i = 0; i < 256; i++) {
		float pi = (float)v[i] / M;
		p[i] = pi;
	}
	return p;
}
void testL3P2() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		vector<int> v = calculateIntensityHistogram(src);
		vector<float> p = FDP(v, height, width);
		imshow("imaginea", src);
		waitKey();
	}
}
vector<int> acumulator(Mat image, int m) {
	vector<int> histogram(m, 0);
	int interval = 255 / m + 1;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			int intensity = image.at<uchar>(i, j);
			int bin = intensity / interval;
			histogram[bin]++;
		}
	}
	return histogram;

}
void testL3P4(int m) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		vector<int> v = acumulator(src, m);
		showHistogram("histogram", v.data(), m, 200);
		imshow("imaginea", src);
		waitKey();
	}

}

void lab4()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		lab4_trasaturi(src);
		waitKey();
	}
}
void lab5bfs() {
	int i, j;
	//queue<pair<int, int>> Q;
	//Q.push(pair<int, int>(i, j));
	//pair<int, int> p = Q.front(); Q.pop();
	vector<vector<int>> edges(1000);
	//dacă u este echivalent cu v
	//edges[u].push_back(v);
	//edges[v].push_back(u);
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat labels(height, width, CV_8UC1,Scalar(0));
		int label = 0;
		int di[8] = { -1,-1,-1,0,0,1,1,1 };
		int dj[8] = { -1,0,1,-1,1,-1,0,1 };
		imshow("src", src);
		for (int i = 1; i < height-1; i++) {
			for (int j = 1; j < width-1; j++) {
				if ((src.at<uchar>(i, j) == 0) && (labels.at<uchar>(i, j) == 0)) {
					label++;
					queue<pair<int, int>> Q;
					labels.at<uchar>(i, j) = label;
					Q.push({i, j});
					while (!Q.empty()) {
						std::pair<int, int> coord = Q.front();
						Q.pop();
						for (int k = 0; k < 8; k++) {
							int ci = coord.first + di[k];
							int cj = coord.second + dj[k];
							if(isInside(src,ci,cj)){
								if ((src.at<uchar>(ci, cj) == 0) && (labels.at<uchar>(ci, cj) == 0)) {
									labels.at<uchar>(ci, cj) = label;
									Q.push({ ci, cj });
								}
							}
							
						}
					}
				}
			}
		}
		imshow("labels", labels);
		Mat dst(src.rows, src.cols, CV_8UC3, Scalar(255, 255, 255));
		std::vector<Vec3b> color1 = { Vec3b(255,255,255) };

		srand(time(NULL));
		for (int k = 1; k <= label; k++) {
			color1.push_back(Vec3b(rand() % 256, rand() % 256, rand() % 256));
		}
		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				if (labels.at<uchar>(i, j) != 0)
					dst.at<Vec3b>(i, j) = color1[labels.at<uchar>(i, j)];
			}
		}

		imshow("color", dst);


		waitKey();
	}
}
void lab5_two_pass()
{
	char fname[MAX_PATH];
	if (!openFileDlg(fname))
		return;
	Mat_<uchar>img = imread(fname, IMREAD_GRAYSCALE);
	Mat dst = Mat(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));
	Mat labels = Mat::zeros(img.rows, img.cols, CV_32SC1);
	std::vector<std::vector<int>> edges = std::vector<std::vector<int>>(500);
	int label = 0;
	int x = 0;
	int y = 0;
	int newLabel = 0;
	int height = img.rows;
	int width = img.cols;
	int di[] = { 0, -1, -1, -1};
	int dj[] = { -1, -1, 0, 1 };
	for (int i = 0; i < height - 1; i++)
	{
		for (int j = 0; j < width - 1; j++)
		{
			if ((img.at<uchar>(i, j) == 0) && (labels.at<int>(i, j) == 0))
			{
				std::vector<int> L;
				for (int d = 0; d < 4; d++)
				{
					if (labels.at<int>(i + di[d], j + dj[d]) > 0)
					{
						L.push_back(labels.at<int>(i + di[d], j + dj[d]));
					}
				}
				if (L.size() == 0)
				{
					label++;
					labels.at<int>(i, j) = label;
				}
				else
				{
					x = *min_element(L.begin(), L.end());
					labels.at<int>(i, j) = x;
					for (int y : L)
					{
						if (y != x)
						{
							edges[x].push_back(y);
							edges[y].push_back(x);
						}
					}
				}
			}
		}
	}
	std::vector<int> newLabels(label + 1);
	for (int i = 1; i <= label; i++)
	{
		if (newLabels[i] == 0)
		{
			newLabel++;
			std::queue <int> q;
			newLabels[i] = newLabel;
			q.push(i);
			while (!q.empty())
			{
				int x = q.front();
				q.pop();
				for (int y : edges[x])
				{
					if (newLabels[y] == 0)
					{
						newLabels[y] = newLabel;
						q.push(y);
					}
				}
			}
		}
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			labels.at<int>(i, j) = newLabels[labels.at<int>(i, j)];
		}
	}

	std::vector<Vec3b> colors;
	for (int i = 0; i <= label; i++) {
		colors.push_back(Vec3b(rand() % 256, rand() % 256, rand() % 256));
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (labels.at<int>(i, j) != 0)
				dst.at<Vec3b>(i, j) = colors[labels.at<int>(i, j)];
		}
	}

	imshow("labels", dst);

	waitKey(0);
}
int calculateDerivata(int dir1, int dir2) {
	int der = (dir2 - dir1 + 8) % 8;
	return der;
}
void lab6_contur() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int di[8] = { 0,-1,-1,-1, 0, 1,1,1 };
		int dj[8] = { 1, 1, 0,-1,-1,-1,0,1 };
			//	      0  1  2  3  4  5 6 7		
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst(height, width, CV_8UC1);
		vector<int> chaincode(1000);
		int derivata[1000],k=0;
		int p1i, p1j, p2i, p2j,p3i,p3j;
		int dir=7;
		int stop = 0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if ((src.at<uchar>(i, j) == 0)) {
					p1i = i;
					p1j = j;
					stop = 1;
					break;
				}
			}
			if (stop == 1)
				break;
		}
		dst.at<uchar>(p1i, p1j)=0;
		p2i = p1i;
		p2j = p1j;
		stop = 1;
		int oldDir;
		do {
			chaincode.push_back(dir);
			printf("%d ", dir);
			oldDir = dir;
			if (dir % 2 == 0)
				dir = (dir + 7) % 8;
			else
				dir = (dir + 6) % 8;
			while (src.at<uchar>(p2i + di[dir], p2j + dj[dir]) != 0) {
				dir = (dir + 1) % 8;
			}

			derivata[k++] = calculateDerivata(oldDir, dir);

			dst.at<uchar>(p2i + di[dir], p2j + dj[dir])=0;
			p2i = p2i + di[dir];
			p2j = p2j + dj[dir];
			if (p2i == p1i && p2j == p1j)
				stop = 0;
		} while (stop==1);
		for (int i = 0; i < k; i++)
			printf("%d ", derivata[i]);
		printf("\n");
		imshow("src", src);
		imshow("dst", dst);
		waitKey();

	}
}
void lab6_reconstructImage() {
	Point P0;
	int n;
	Mat dest;
	dest = imread("Images//gray_background.bmp", IMREAD_GRAYSCALE);

	std::ifstream fin("Resources/reconstruct.txt");
	fin >> P0.x >> P0.y;
	fin >> n;

	Point newP = P0;
	int dir[][2] = { { 0, 1 }, { -1, 1 }, { -1, 0 }, { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 } };
	dest.at<uchar>(newP.x, newP.y) = 0;
	while (n--) {
		int currDir;
		fin >> currDir;

		newP = { newP.x + dir[currDir][0], newP.y + dir[currDir][1] };
		dest.at<uchar>(newP.x, newP.y) = 0;
	}

	imshow("Reconstructed image", dest);
	waitKey();
}

Mat dilataree_8(Mat src) {
	imshow("src", src);
	int height = src.rows;
	int width = src.cols;
	Mat dilatare(height, width, CV_8UC1);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 0) {
				dilatare.at<uchar>(i, j) = 0;
				if (isInside(src, i - 1, j))
					dilatare.at<uchar>(i - 1, j) = 0;
				if (isInside(src, i - 1, j-1))
					dilatare.at<uchar>(i - 1, j-1) = 0;
				if (isInside(src, i - 1, j+1))
					dilatare.at<uchar>(i - 1, j+1) = 0;
				if (isInside(src, i + 1, j))
					dilatare.at<uchar>(i + 1, j) = 0;
				if (isInside(src, i + 1, j-1))
					dilatare.at<uchar>(i + 1, j-1) = 0;
				if (isInside(src, i + 1, j+1))
					dilatare.at<uchar>(i + 1, j+1) = 0;
				if (isInside(src, i, j - 1))
					dilatare.at<uchar>(i, j - 1) = 0;
				if (isInside(src, i, j + 1))
					dilatare.at<uchar>(i, j + 1) = 0;
			}
		}
	}
	return dilatare;
}
Mat dilataree_4(Mat src) {
	imshow("src", src);
	int height = src.rows;
	int width = src.cols;
	Mat dilatare(height, width, CV_8UC1);
	for (int i = 1; i < height-1; i++) {
		for (int j = 1; j < width-1; j++) {
			if (src.at<uchar>(i, j) == 0) {
				dilatare.at<uchar>(i, j) = 0;
				if (isInside(src, i - 1, j))
					dilatare.at<uchar>(i - 1, j) = 0;
				if (isInside(src, i + 1, j))
					dilatare.at<uchar>(i + 1, j) = 0;
				if (isInside(src, i, j - 1))
					dilatare.at<uchar>(i, j - 1) = 0;
				if (isInside(src, i, j + 1))
					dilatare.at<uchar>(i, j + 1) = 0;
			}
		}
	}
	return dilatare;
}
void lab7_dilatare() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dilatare = dilataree_4(src);
		imshow("dilatare", dilatare);
		waitKey();
	}
}
Mat eroziune_8(Mat src) {
	int height = src.rows;
	int width = src.cols;
	Mat eroziune(height, width, CV_8UC1);
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (src.at<uchar>(i, j) == 0) {
				if (isInside(src, i - 1, j) && isInside(src, i + 1, j) && isInside(src, i - 1, j - 1) && isInside(src, i - 1, j + 1) && isInside(src, i + 1, j - 1) && isInside(src, i+1, j+1) && isInside(src, i, j - 1) && isInside(src, i, j + 1))
				{
					if (src.at<uchar>(i - 1, j) == 0
						&& src.at<uchar>(i - 1, j - 1) == 0
						&& src.at<uchar>(i - 1, j + 1) == 0
						&& src.at<uchar>(i + 1, j) == 0
						&& src.at<uchar>(i + 1, j + 1) == 0
						&& src.at<uchar>(i + 1, j - 1) == 0
						&& src.at<uchar>(i, j - 1) == 0
						&& src.at<uchar>(i, j + 1) == 0)
					{
						eroziune.at<uchar>(i, j) = 0;
					}
				}
			}
		}
	}
	return eroziune;
}
Mat eroziunee_4(Mat src) {
	int height = src.rows;
	int width = src.cols;
	Mat eroziune(height, width, CV_8UC1);
	for (int i = 1; i < height-1; i++) {
		for (int j = 1; j < width-1; j++) {
			if (src.at<uchar>(i, j) == 0) {
				if (isInside(src, i - 1, j) && isInside(src, i + 1, j) && isInside(src, i, j - 1) && isInside(src, i, j + 1))
				{
					if (src.at<uchar>(i - 1, j) == 0 
						&& src.at<uchar>(i + 1, j) == 0
						&& src.at<uchar>(i, j - 1) == 0
						&& src.at<uchar>(i, j + 1) == 0)
					{
						eroziune.at<uchar>(i, j) = 0;
					}
				}
			}
		}
	}
	return eroziune;
}
void lab7_eroziune() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("src", src);
		Mat eroziune = eroziunee_4(src);
		imshow("eroziune", eroziune);
		waitKey();
	}
}
Mat deschidere(Mat src){
	int height = src.rows;
	int width = src.cols;
	Mat dst(height, width, CV_8UC1);
	dst = eroziunee_4(src);
	dst = dilataree_4(dst);
	return dst;
}
Mat inchidere(Mat src) {
	int height = src.rows;
	int width = src.cols;
	Mat dst(height, width, CV_8UC1);
	dst = dilataree_4(src);
	dst = eroziunee_4(dst);
	return dst;
}
void lab7_deschidere_inchidere() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("src", src);
		Mat matDeschis = deschidere(src);
		Mat matInchis = inchidere(src);
		imshow("Deschidere", matDeschis);
		imshow("Inchidere", matInchis);
		waitKey();
	}
}
//complement
Mat notMat(Mat src) {
	Mat dst = Mat(src.rows,src.cols,CV_8UC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0)
				dst.at<uchar>(i, j) = 255;
			else
				dst.at<uchar>(i, j) = 0;
		}
	}
	return dst;
}
Mat intersectie(Mat src1, Mat src2) {
	int height = src1.rows;
	int width = src1.cols;
	Mat intersectiee = Mat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src1.at<uchar>(i, j) == src2.at<uchar>(i, j))
				intersectiee.at<uchar>(i, j) = src2.at<uchar>(i, j);
			else intersectiee.at<uchar>(i, j) = 255;
		}
	}
	return intersectiee;
}
Mat reuniune(Mat src1, Mat src2) {
	int height = src1.rows;
	int width = src1.cols;
	Mat reuniunee = Mat(height, width, CV_8UC1,Scalar(255));
	for (int i = 0; i < src1.rows && i<src2.rows; i++) {
		for (int j = 0; j < src1.cols && j<src2.cols; j++) {
			if (src1.at<uchar>(i, j)==0)
				reuniunee.at<uchar>(i, j) = 0;
			else if(src2.at<uchar>(i, j)==0)
				reuniunee.at<uchar>(i, j) = 0;
			else reuniunee.at<uchar>(i, j) = 255;
		}
	}
	return reuniunee;
}
void lab7_extragere_contur_4() {
	int di[4] = { -1,0,1,0 };
	int dj[4] = { 0,-1,0,1 };
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat srcAerozB = Mat(height, width, CV_8UC1);
		imshow("src", src);
		srcAerozB = eroziunee_4(src);
		srcAerozB = notMat(srcAerozB);
		Mat conturA = Mat(height, width, CV_8UC1,Scalar(0));
		conturA = intersectie(src, srcAerozB);
		//conturA = notMat(conturA);
		//imshow("srcAeroxB", srcAerozB);
		imshow("conturA", conturA);
		waitKey();
	}
}
void lab7_umplerea_regiunilor_4() {
	int di[4] = { -1,0,1,0 };
	int dj[4] = { 0,-1,0,1 };
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Point p;
		p.x = height / 2;
		p.y = width / 2;
		Mat B = Mat(height, width, CV_8UC1, Scalar(255));
		Mat oldXk = Mat(height, width, CV_8UC1, Scalar(255));
		Mat newXk = Mat(height, width, CV_8UC1);
		Mat Ac = notMat(src);
		uchar op, op1;
		oldXk.at<uchar>(p.x, p.y) = 0;
		bool egal = false;
		imshow("Ac", Ac);
		do {
			for (int i = 1; i < height - 1; i++)
			{
				for (int j = 1; j < width - 1; j++)
				{
					if (oldXk.at<uchar>(i, j) == 0) {
						B.at<uchar>(i, j) = 0;
						for (int k = 0; k < 4; k++) {
							B.at<uchar>(i + di[k], j + dj[k]) = 0;
						}
					}

				}
			}
			newXk = intersectie(Ac, B);
			egal = true;
			for (int i = 1; i < height - 1; i++)
			{
				for (int j = 1; j < width - 1; j++)
				{
					if (newXk.at<uchar>(i, j) != oldXk.at<uchar>(i, j))
					{
						egal = false;
						j = width;
						i = height;
					}
				}
			}
			newXk.copyTo(oldXk);
		} while (!egal);


		imshow("src", src);
		imshow("output image", newXk);
		waitKey();
	}
}
void lab7_reuniune() {
	//char fname[MAX_PATH];
	//while (openFileDlg(fname))
	//{
		Mat src = imread("Images//shapes.bmp", IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat src1;
		// 	dest = imread("Images//gray_background.bmp", IMREAD_GRAYSCALE);

		src1=imread("Images//wdg2thr3_bw.bmp", IMREAD_GRAYSCALE);
		imshow("src", src);
		imshow("src1", src1);
		Mat reuniunee= Mat(height, width, CV_8UC1);
		reuniunee = reuniune(src, src1);
		//conturA = notMat(conturA);
		//imshow("srcAeroxB", srcAerozB);
		imshow("reuniune", reuniunee);
		waitKey();
	//}
}

void lab8_media_deviataia_histograma_cumulativa() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		uchar BLACK = 0;
		uchar WHITE = 255;
		int height = src.rows;
		int width = src.cols;
		int histogram[255] = { 0 };
		double intensityMedie = 0.0,devstd=0.0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int intensity = src.at<uchar>(i, j);
				histogram[intensity]++;
				intensityMedie += src.at<uchar>(i, j);
			}
		}
		intensityMedie = intensityMedie / (height * width);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int intensity = src.at<uchar>(i, j);
				devstd += (intensity - intensityMedie) * (intensity - intensityMedie);
			}
		}
		devstd = sqrt(devstd / (height * width));
		showHistogram("Histogram", histogram, 256, 512);
		imshow("Src", src);
		int cumulativeHistogram[255] = { 0 };
		cumulativeHistogram[0] = histogram[0];
		for (int i = 1; i < 255; ++i) {
			cumulativeHistogram[i] = cumulativeHistogram[i - 1] + histogram[i];
		}
		showHistogram("Cumulative Histogram", cumulativeHistogram, 255, 510);
		waitKey(0);
	}
}
void lab8_binarizare_automata()  {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		uchar BLACK = 0;
		uchar WHITE = 255;
		int height = src.rows;
		int width = src.cols;
		int histogram[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int intensity = src.at<uchar>(i, j);
				histogram[intensity]++;
			}
		}
		int imin=0, imax=255;
		while (histogram[imax] == 0) imax--;
		while (histogram[imin] == 0) imin++;

		cout << imax << " " << imin << "\n";
		double T = (double)(imin + imax) / 2;
		double terror = 0.1;
		double previousT;
		do {
			previousT = T;

			double uG1 = 0, uG2 = 0;
			int nG1 = 0, nG2 = 0;
			for (int i = imin; i <= T; i++) {
				uG1 += i * histogram[i];
				nG1 += histogram[i];
			}
			for (int i = T + 1; i <= imax; i++) {
				uG2 += i * histogram[i];
				nG2 += histogram[i];
			}
			if (uG1 != 0) uG1 /= nG1;
			if (nG2 != 0) uG2 /= nG2;
			
			T = ((uG1 + uG2) / 2);

		} while (abs(T - previousT) > terror);
		cout << "T:" << T;
		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) < T)
					dst.at<uchar>(i, j) = 0;
				else dst.at<uchar>(i, j) = 255;
			}
		}
		imshow("src",src);
		imshow("dst", dst);
		waitKey(0);
	}
}
double* computeFDPC(const Mat& src) {
	double* fdpc = new double[256] {0.0};

	int height = src.rows;
	int width = src.cols;
	int totalPixels = height * width;

	vector <int> histogram = calculateIntensityHistogram(src);

	for (int i = 0; i < 256; i++) {
		fdpc[i] = static_cast<double>(histogram[i]) / totalPixels;
	}

	return fdpc;
}
void lab8_transformare_analitica() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		uchar BLACK = 0;
		uchar WHITE = 255;
		int height = src.rows;
		int width = src.cols;
		int histogram[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int intensity = src.at<uchar>(i, j);
				histogram[intensity]++;
			}
		}
		int histogramNegativ[256] = { 0 };
		Mat srcNegativ = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for(int j=0;j<width;j++ ){
				int negIntensity = 255-src.at<uchar>(i, j);
				srcNegativ.at<uchar>(i, j)=negIntensity;
				histogram[negIntensity]++;
			}
		}
		int offset=-50;
		Mat dstLuminozitate = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int intensity = src.at<uchar>(i, j);
				if (intensity + offset < 0)
					dstLuminozitate.at<uchar>(i, j) = 0;
				else if (intensity + offset > 255)
					dstLuminozitate.at<uchar>(i, j) = 255;
				else dstLuminozitate.at<uchar>(i, j) = intensity + offset;
			}
		}
		Mat contrast = Mat(height, width, CV_8UC1);
		double goutMin = 10;
		double goutMax = 150;
		int ginMin=0, ginMax = 255;
		while (histogram[ginMin] == 0) ginMin++;
		while (histogram[ginMax] == 0) ginMax--;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int gin = src.at<uchar>(i, j);
				int gout = goutMin + ((gin - ginMin) * (goutMax - goutMin)) / (ginMax - ginMin);
				contrast.at<uchar>(i, j) = gout;
			}
		}
		Mat gamma = Mat(height, width, CV_8UC1);
		double gammaValue = 0.02;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				double in = src.at<uchar>(i, j);
				int out = 255 * pow((in / 255.0), gammaValue);
				gamma.at<uchar>(i, j) = out;
			}
		}
		Mat equalization = Mat(height, width, CV_8UC1);
		double* pdf = computeFDPC(src);
		double cdf[256] = {};
		cdf[0] = pdf[0];
		for (int i = 1; i < 256; i++) {
			cdf[i] = cdf[i - 1] + pdf[i];
		}
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int in = src.at<uchar>(i, j);
				int out = 255 * cdf[in];
				equalization.at<uchar>(i, j) = out;
			}
		}

		imshow("src", src);
		imshow("srcNegativ", srcNegativ);
		imshow("dstLuminozitate", dstLuminozitate);
		imshow("contrast", contrast);
		imshow("equalization", equalization);
		imshow("gamma", gamma);
		waitKey(0);
	}
}

void lab9_mediaAritmeticaFilter() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		int H[3][3] = { 1,1,1,1,1,1,1,1,1 };
		int divide = 9;
		int w = 3;

		Mat dst = Mat(height, width, CV_8UC1);

		dst = src.clone();

		for (int i = 1; i < src.rows-1; i++) {
			for (int j = 1; j < src.cols-1; j++) {
				float pixelVal = 0;
				for (int k = 0; k < w; k++) {
					for (int l = 0; l < w; l++) {
						pixelVal += H[k][l] * src.at<uchar>(i + k - (w / 2), j + l - (w / 2));
					}
				}
				dst.at<uchar>(i, j) = pixelVal / divide;

			}
		}

		imshow("src", src);
		imshow("Result", dst);
		waitKey(0);
	}
}
void lab9_gaussianFilter() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		int H[3][3] = { 1,2,1,2,4,2,1,2,1 };
		int divide = 16;
		int w = 3;

		Mat dst = Mat(height, width, CV_8UC1);

		dst = src.clone();

		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				float pixelVal = 0;
				for (int k = 0; k < w; k++) {
					for (int l = 0; l < w; l++) {
						pixelVal += H[k][l] * src.at<uchar>(i + k - (w / 2), j + l - (w / 2));
					}
				}
				dst.at<uchar>(i, j) = pixelVal / divide;

			}
		}

		imshow("src", src);
		imshow("Result", dst);
		waitKey(0);
	}
}
void lab9_laplaceFilter() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		int H[3][3] = { 0,-1,0,-1,4,-1,0,-1,0 };
		int w = 3;

		Mat dst = Mat(height, width, CV_8UC1);

		dst = src.clone();

		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				float pixelVal = 0;
				for (int k = 0; k < w; k++) {
					for (int l = 0; l < w; l++) {
						pixelVal += H[k][l] * src.at<uchar>(i + k - (w / static_cast<float>(2)), j + l - (w / static_cast<float>(2)));
					}
				}
				if (pixelVal < 0) {
					dst.at<uchar>(i, j) = 0;
				}
				else if (pixelVal > 255) {
					dst.at<uchar>(i, j) = 255;
				}
				else dst.at<uchar>(i, j) = pixelVal;

			}
		}

		imshow("src", src);
		imshow("Result", dst);
		waitKey(0);
	}
}
void lab9_highPassFilter() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		int H[3][3] = { 0,-1,0,-1,5,-1,0,-1,0 };
		int w = 3;

		Mat dst = Mat(height, width, CV_8UC1);

		dst = src.clone();

		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				float pixelVal = 0;
				for (int k = 0; k < w; k++) {
					for (int l = 0; l < w; l++) {
						pixelVal += H[k][l] * src.at<uchar>(i + k - (w / 2), j + l - (w / 2));
					}
				}
				if (pixelVal < 0) {
					dst.at<uchar>(i, j) = 0;
				}
				else if (pixelVal > 255) {
					dst.at<uchar>(i, j) = 255;
				}
				else dst.at<uchar>(i, j) = pixelVal;
			}
		}

		imshow("src", src);
		imshow("Result", dst);
		waitKey(0);
	}
}

void lab10_medianfilter() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		int w = 3;

		Mat dst = Mat(height, width, CV_8UC1);

		dst = src.clone();

		for (int i = w/2; i < src.rows-w/2-1 ; i++) {
			for (int j = w/2; j < src.cols-w/2-1; j++) {
				std::vector<uchar> contur;
				contur.clear();
				for (int k = 0; k < w; k++) {
					for (int l = 0; l < w; l++) {
						int ni = i + k - (w / 2);
						int nj = j + l - (w / 2);
						if (isInside(src, ni, nj)) {
							contur.push_back(src.at<uchar>(ni, nj));
						}
					}
				}
				if (!contur.empty()) {
					std::sort(contur.begin(), contur.end());
					dst.at<uchar>(i, j) = contur[contur.size() / 2];
				}
				
			}
		}

		imshow("src", src);
		imshow("Result", dst);
		waitKey(0);
	}
}

void lab10_gaussian1x2d() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int m = height * width;
		float v = 0.8;
		int w = ceil(v * 6); //5
		cout << w;
		
		Mat dst = Mat(height, width, CV_8UC1);
		dst = src.clone();
		Mat tmp = Mat(height, width, CV_8UC1);
		if (w % 2 == 0) {
			w++;
		}

		int x0 = w / 2, y0 = w / 2;
		float G[10][10];
		float sum = 0;
		float constant = 1 / (2 * PI * v * v);

		for (int i = 0; i < w; i++) {
			for (int j = 0; j < w; j++) {
				G[i][j] = constant * exp(-((i - x0) * (i - x0) + (j - y0) * (j - y0)) / (2 * v * v));
				sum = sum + G[i][j];
			}
		}
		for (int i = w/2; i < src.rows-w/2-1 ; i++) {
			for (int j = w/2; j < src.cols-w/2-1; j++) {
				float res = 0;
				for (int k = 0; k < w; k++) {
					for (int l = 0; l < w; l++) {
						int ni = i + k - (w / 2);
						int nj = j + l - (w / 2);
						if (isInside(src, ni, nj)) {
							res=res+(src.at<uchar>(ni, nj))*G[k][l];
						}
					}
				}
				res = res / sum;
				if (res > 255) {
					dst.at<uchar>(i, j) = 255;

				}
				else if (res < 0) {
					dst.at<uchar>(i, j) = 0;

				}
				else {
					dst.at<uchar>(i, j) = res;
				}
			}
		}
		
		imshow("src", src);
		imshow("Result", dst);
		waitKey(0);
	}
}

void lab10_gaussian2x1d() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int m = height * width;
		float v = 0.8;
		int w = ceil(v * 6); //5
		cout << w;

		Mat dst = Mat(height, width, CV_8UC1);
		dst = src.clone();
		Mat tmp = Mat(height, width, CV_8UC1);
		if (w % 2 == 0) {
			w++;
		}

		int x0 = w / 2, y0 = w / 2;
		float Gi[10],Gj[10];
		float sum = 0;
		float constant = 1 / sqrt(2 * PI * v * v);
		for (int i = 0; i < w; i++) {
			Gi[i] = constant * exp(-((i - x0) * (i - x0)) / (2 * v * v));
			sum = sum + Gi[i];
		}
		
		for (int i = w/2; i < src.rows-w/2-1; i++) {
			for (int j = w/2; j < src.cols-w/2-1; j++) {
				float res = 0;
				for (int k = 0; k < w; k++) {
					int ni = i + k - (w / 2);
					if (isInside(src, ni, j)) {
						res = res + (src.at<uchar>(ni, j)) * Gi[k];
					}
				}
				res = res / sum;
				tmp.at<uchar>(i, j) = res;
			}
		}
		sum = 0;
		
		for (int j = 0; j < w; j++) {
			Gj[j] = constant * exp(-((j - y0) * (j - y0)) / (2 * v * v));
			sum = sum + Gj[j];
		}

		for (int i = w/2; i < tmp.rows-w/2-1; i++) {
			for (int j = w/2; j < tmp.cols-w/2-1; j++) {
				float res = 0;
				for (int k = 0; k < w; k++) {
					int nj = j + k - (w / 2);
					if (isInside(tmp, k, nj)) {
						res = res + (tmp.at<uchar>(i, nj)) * Gj[k];
					}
				}
				res = res / sum;
				dst.at<uchar>(i, j) = res;

			}
		}

		imshow("src", src);
		imshow("Result", dst);
		waitKey(0);
	}
}

void lab11_roberts() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("src", src);
		int height = src.rows;
		int width = src.cols;

		float Gx[2][2] = { {1, 0}, {0, -1} };
		float Gy[2][2] = { {0, 1}, {-1, 0} };
		Mat dstX = Mat::zeros(height, width, CV_32FC1);
		Mat dstY = Mat::zeros(height, width, CV_32FC1);

		//  gradient
		for (int i = 0; i < height - 1; i++) {
			for (int j = 0; j < width - 1; j++) {
				float resX = 0;
				float resY = 0;
				for (int k = 0; k < 2; k++) {
					for (int l = 0; l < 2; l++) {
						resX += Gx[k][l] * src.at<uchar>(i + k, j + l);
						resY += Gy[k][l] * src.at<uchar>(i + k, j + l);
					}
				}
				dstX.at<float>(i, j) = resX;
				dstY.at<float>(i, j) = resY;
			}
		}

		Mat modul_float = Mat(height, width, CV_32FC1);
		Mat directie = Mat(height, width, CV_32FC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				modul_float.at<float>(i, j) = sqrt(dstX.at<float>(i, j) * dstX.at<float>(i, j) + dstY.at<float>(i, j) * dstY.at<float>(i, j));
				directie.at<float>(i, j) = atan2(dstY.at<float>(i, j), dstX.at<float>(i, j));
			}
		}

		Mat modul = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				modul.at<uchar>(i, j) = saturate_cast<uchar>(modul_float.at<float>(i, j));
			}
		}

		imshow("modul", modul);
		waitKey(0);
	}
}

void lab11_sobel() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("src", src);
		int height = src.rows;
		int width = src.cols;
		int w = 3;
		// Sobel
		float Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
		float Gy[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };
		Mat dstX = Mat::zeros(height, width, CV_32FC1);
		Mat dstY = Mat::zeros(height, width, CV_32FC1);

		// Calculate gradient
		for (int i = w / 2; i < height - w / 2; i++) {
			for (int j = w / 2; j < width - w / 2; j++) {
				float resX = 0;
				float resY = 0;
				for (int k = 0; k < w; k++) {
					for (int l = 0; l < w; l++) {
						resX += Gx[k][l] * src.at<uchar>(i + k - w / 2, j + l - w / 2);
						resY += Gy[k][l] * src.at<uchar>(i + k - w / 2, j + l - w / 2);
					}
				}
				dstX.at<float>(i, j) = resX;
				dstY.at<float>(i, j) = resY;
			}
		}

		Mat modul_float = Mat(height, width, CV_32FC1);
		Mat directie = Mat(height, width, CV_32FC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				modul_float.at<float>(i, j) = sqrt(dstX.at<float>(i, j) * dstX.at<float>(i, j) + dstY.at<float>(i, j) * dstY.at<float>(i, j));
				directie.at<float>(i, j) = atan2(dstY.at<float>(i, j), dstX.at<float>(i, j));
			}
		}

		Mat modul = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				modul.at<uchar>(i, j) = saturate_cast<uchar>(modul_float.at<float>(i, j));
			}
		}

		imshow("modul", modul);
		waitKey(0);
	}
}

void lab11_prewitt() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("src", src);
		int height = src.rows;
		int width = src.cols;
		//Prewitt
		float Gx[3][3] = { -1,0,1,-1,0,1,-1,0,1 };
		float Gy[3][3] = { 1,1,1,0,0,0,-1,-1,-1 };
		Mat dstX = Mat::zeros(height, width, CV_32FC1);
		Mat dstY = Mat::zeros(height, width, CV_32FC1);
		int w = 3;
		// gradientul
		for (int i = w / 2; i < height - w / 2; i++) {
			for (int j = w / 2; j < width - w / 2; j++) {
				float resX = 0;
				float resY = 0;
				for (int k = 0; k < w; k++) {
					for (int l = 0; l < w; l++) {
						resX += Gx[k][l] * src.at<uchar>(i + k - w / 2, j + l - w / 2);
						resY += Gy[k][l] * src.at<uchar>(i + k - w / 2, j + l - w / 2);
					}
				}
				dstX.at<float>(i, j) = resX;
				dstY.at<float>(i, j) = resY;
			}
		}

		Mat modul_float = Mat(height, width, CV_32FC1);
		Mat directie = Mat(height, width, CV_32FC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				modul_float.at<float>(i, j) = sqrt(dstX.at<float>(i, j) * dstX.at<float>(i, j) + dstY.at<float>(i, j) * dstY.at<float>(i, j));
				directie.at<float>(i, j) = atan2(dstY.at<float>(i, j), dstX.at<float>(i, j));
			}
		}

		Mat modul = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				modul.at<uchar>(i, j) = saturate_cast<uchar>(modul_float.at<float>(i, j));
			}
		}
		imshow("src", src);
		imshow("modul", modul);
		waitKey(0);
	}
}

void lab11_canny() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("src", src);
		Mat tmp = src.clone();
		int height = src.rows;
		int width = src.cols;
		int w = 3;
		float v = w / 6.0;
		int x0 = w / 2;
		int y0 = w / 2;
		float constant = 1 / (2.0 * v * v * CV_PI);
		float G[3][3];
		float sum = 0;

		//filtrul Gaussian
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < w; j++) {
				G[i][j] = constant * exp(-((i - x0) * (i - x0) + (j - y0) * (j - y0)) / (2 * v * v));
				sum += G[i][j];
			}
		}

		// Normalizare
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < w; j++) {
				G[i][j] /= sum;
			}
		}

		// filtram
		for (int i = w / 2; i < height - w / 2; i++) {
			for (int j = w / 2; j < width - w / 2; j++) {
				float res = 0;
				for (int k = 0; k < w; k++) {
					for (int l = 0; l < w; l++) {
						res += G[k][l] * src.at<uchar>(i + k - w / 2, j + l - w / 2);
					}
				}
				tmp.at<uchar>(i, j) = saturate_cast<uchar>(res);
			}
		}

		imshow("filtru gaussian", tmp);

		//Prewitt
		float Gx[3][3] = { -1,0,1,-1,0,1,-1,0,1 };
		float Gy[3][3] = { 1,1,1,0,0,0,-1,-1,-1 };
		//Sobel
		//float Gx[3][3] = {-1,0,1,-2,0,2,-1,0,1};
		//float Gy[3][3] = {1,2,1,0,0,0,-1,-2,-1};

		Mat dstX = Mat::zeros(height, width, CV_32FC1);
		Mat dstY = Mat::zeros(height, width, CV_32FC1);

		// gradientul
		for (int i = w / 2; i < height - w / 2; i++) {
			for (int j = w / 2; j < width - w / 2; j++) {
				float resX = 0;
				float resY = 0;
				for (int k = 0; k < w; k++) {
					for (int l = 0; l < w; l++) {
						resX += Gx[k][l] * tmp.at<uchar>(i + k - w / 2, j + l - w / 2);
						resY += Gy[k][l] * tmp.at<uchar>(i + k - w / 2, j + l - w / 2);
					}
				}
				dstX.at<float>(i, j) = resX;
				dstY.at<float>(i, j) = resY;
			}
		}

		Mat modul_float = Mat(height, width, CV_32FC1);
		Mat directie = Mat(height, width, CV_32FC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				modul_float.at<float>(i, j) = sqrt(dstX.at<float>(i, j) * dstX.at<float>(i, j) + dstY.at<float>(i, j) * dstY.at<float>(i, j));
				directie.at<float>(i, j) = atan2(dstY.at<float>(i, j), dstX.at<float>(i, j));
			}
		}

		Mat modul = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				modul.at<uchar>(i, j) = saturate_cast<uchar>(modul_float.at<float>(i, j));
			}
		}

		/*
		//pt Sobel
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				modul.at<float>(i, j) = modul.at<float>(i, j) / (4 * sqrt(2));
			}
		}
		*/

		//determinam muchiile 
		Mat non_maxima = modul.clone();
		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				int direction = 0;
				float angle = directie.at<float>(i, j) * 180.0f / CV_PI;
				if (angle < 0.0f) {
					angle += 360.0f;
				}
				if ((angle >= 337.5f) || (angle < 22.5f)|| (angle >= 157.5f && angle < 202.5f)) {
					direction = 2;
				}
				else if ((angle >= 22.5 && angle < 67.5f) || (angle >= 202.5f && angle < 247.5f)) {
					direction = 1;
				}
				else if ((angle >= 112.5f && angle < 157.5f)|| (angle >= 292.5f && angle < 337.5f)) {
					direction = 3;
				}
				bool max = true;
				if (direction == 0) {
					if (isInside(src, i - 1, j) && modul.at<uchar>(i, j) < modul.at<uchar>(i - 1, j)) {
						max = false;
					}
					if (isInside(src, i + 1, j) && modul.at<uchar>(i, j) < modul.at<uchar>(i + 1, j)) {
						max = false;
					}
				}
				else if (direction == 1) {
					if (isInside(src, i - 1, j + 1) && modul.at<uchar>(i, j) < modul.at<uchar>(i - 1, j + 1)) {
						max = false;
					}
					if (isInside(src, i + 1, j - 1) && modul.at<uchar>(i, j) < modul.at<uchar>(i + 1, j - 1)) {
						max = false;
					}
				}
				else if (direction == 2) {
					if (isInside(src, i, j + 1) && modul.at<uchar>(i, j) < modul.at<uchar>(i, j + 1)) {
						max = false;
					}
					if (isInside(src, i, j - 1) && modul.at<uchar>(i, j) < modul.at<uchar>(i, j - 1)) {
						max = false;
					}
				}
				else if (direction == 3) {
					if (isInside(src, i + 1, j + 1) && modul.at<uchar>(i, j) < modul.at<uchar>(i + 1, j + 1)) {
						max = false;
					}
					if (isInside(src, i - 1, j - 1) && modul.at<uchar>(i, j) < modul.at<uchar>(i - 1, j - 1)) {
						max = false;
					}
				}
				if (!max) {
					non_maxima.at<uchar>(i, j) = 0;
				}
			}
		}

		
		imshow("suprimarea non-maxima", non_maxima);

		// histograma si histerizarea
		int histogram[256] = { 0 };
		int pixelConsiderated = 0;
		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				int value = non_maxima.at<uchar>(i, j);
				histogram[value]++;
				pixelConsiderated++;
			}
		}
		float p = 0.1;
		int nonEdge = static_cast<int>((1.0 - p) * (pixelConsiderated - histogram[0]));
		int currentNo = 0;
		int index = 1;
		while (currentNo <= nonEdge) {
			currentNo += histogram[index];
			index++;
		}
		int prag_high = index - 1;
		float k_prag = 0.4;
		cout << "pragul Adaptiv " << prag_high << "\n"; //prag inalt
		int prag_low = static_cast<int>(prag_high * k_prag);

		Mat mat_histerizare = non_maxima.clone();
		// muchiile tari si slabe
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (non_maxima.at<uchar>(i, j) >= prag_high) {
					mat_histerizare.at<uchar>(i, j) = 255;
				}
				else if (non_maxima.at<uchar>(i, j) >= prag_low) {
					mat_histerizare.at<uchar>(i, j) = 128;
				}
				else {
					mat_histerizare.at<uchar>(i, j) = 0;
				}
			}
		}

		imshow("muchii inainte", mat_histerizare);

		Mat visitat = Mat::zeros(height, width, CV_8UC1);
		Mat muchii_final = mat_histerizare.clone();

		// facem muchiile slabe tari
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (muchii_final.at<uchar>(i, j) == 255 && visitat.at<uchar>(i, j) == 0) {
					queue<pair<int, int>> Q;
					Q.push(make_pair(i, j));
					while (!Q.empty()) {
						pair<int, int> p = Q.front();
						Q.pop();
						visitat.at<uchar>(p.first, p.second) = 255;
						//prin vecini
						int di[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
						int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
						for (int k = 0; k < 8; k++) {
							int ni = p.first + di[k];
							int nj = p.second + dj[k];
							if (isInside(src, ni, nj) && muchii_final.at<uchar>(ni, nj) == 128) {
								Q.push(make_pair(ni, nj));
								muchii_final.at<uchar>(ni, nj) = 255;
							}
						}
					}
				}
			}
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (muchii_final.at<uchar>(i, j) < 255) {
					muchii_final.at<uchar>(i, j) = 0;
				}
			}
		}

		imshow("muchii finale", muchii_final);
		waitKey(0);
	}
}


/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/



int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf("10 - Schimb nivelele de gri cu un factor aditiv \n");
		printf("11 - Schimb nivelele de gri cu un factor multiplicativ \n");
		// Creați o imagine color de dimensiune 256 x 256. Împărțiți imaginea în 4 cadrane egale, și
		//colorați acestea, din stânga - sus până în dreapta - jos, astfel: alb, roșu, verde, galben.
		printf("12 - Problema 5 \n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				int n;
				cin >> n;
				testSchimbNivGriFactorAditiv(n);
				break;
			case 11:
				int m;
				cin >> m;
				testSchimbNivGriFactorMultiplicativ(m);
				break;
			case 12:
				testProblema5Lab1();
				break;
			case 13:
				break;
			case 14:
				testL2P1();
				break;
			case 15:
				testL2P2();
				break;
			case 16:
				int l2n;
				cin >> l2n;
				testL2P3(l2n);
				break;
			case 17:
				testL2P4();
				break;
			case 18:
				int l2i, l2j;
				cin >> l2i >> l2j;
				testL2P5(l2i,l2j);
				break;
			case 19:
				testL3P1();
				break;
			case 20:
				testL3P2();
				break;
			case 21:
				int xL3;
				cin >> xL3;
				testL3P4(xL3);
				break;
			case 22:
				//lab4_1();
				lab4();
				break;
			case 23:
				lab5bfs();
				//labelingBFS();
				break;
			case 24:
				lab6_contur();
				//lab6_reconstructImage();
				
				break;
			case 25:
				//printf("am ajuns aici ce bine");
				lab7_dilatare();
				break;
			case 26:
				//printf("am ajuns aici ce bine");
				lab7_eroziune();
				break;
			case 27:
				//printf("am ajuns aici ce bine");
				lab7_deschidere_inchidere();
				break;
			case 28:
				//lab7_boundary_extraction();
				lab7_extragere_contur_4();
				break;
			case 29:
				lab7_umplerea_regiunilor_4();
				break;
			case 30:
				lab7_reuniune();
				break;
			case 31:
		
				break;
			case 32:
				lab8_media_deviataia_histograma_cumulativa();
				break;
			case 33:
				lab8_binarizare_automata();
				break;
			case 34:
				lab8_transformare_analitica();
				break;
			case 35:
				lab10_medianfilter();
				break;
			case 36:
				lab10_gaussian1x2d();
				break;
			case 37:
				lab10_gaussian2x1d();
				
				break;
			case 38:
				lab11_canny();
				break;
			
		}
	}
	while (op!=0);
	return 0;
}
