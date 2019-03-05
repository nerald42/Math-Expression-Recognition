/** Main source code of the Math Expression Recognition project.

@author Neal Wu (Wu Songji)
@version none (VCS not initiated)
 */

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

/**	@brief Parses standard images their corresponding lables respectively into a dataset.

The function requires the absolute path to the folder storing the standard images of some
glyphs within a math font and parses all the images' grayscale information into a single
binary data set. The images' corresponding lables will be parsed into another data set.
The format of the data set refers to the MNIST database's.

@note Refer to the README file for the requirements about gathering standard images.
 */
void processDataSet();

/** @brief Loads the data sets of standard images and lables.

@param trainImgs Mat object (CV_32FC1) that stores all the bit informationt of the standard images. Each
image per row.
@param trainLables Mat object (CV_32SC1) that stores the corresponding lable per row.
@param rowsNum Number of rows as well as height of a image, which is useful in padding the splited source images.
@param colsNum Number of columns as well as width of a image, which is useful in padding the splited source images.
 */
void loadDataSet(Mat &trainImgs, Mat &trainLables, int &rowsNum, int &colsNum);

/** @brief Preprocesses the source image with a series of methods.

@param mathRawImg Originally input source image.

The fuction preprocess the source image which is not suitable for recognition and transform it into 
a binary image with only useful and correct bit information as possible. These methods in order are:
@code
	blur(...); // Eliminate the pepper and salt noise
	threshold(...); // Binarized the image with Otsu method
	Canny(...); HoughLines(...); warpAffine(...); // Rotate the image according math expression's horizontal direction
	morphologyEX(...); // Use open and close morphologic operation to eliminate the useless tiny connections and cracks.
@endcode
 */
void mathImgPreprocess(Mat &mathRawImg);

/** @brief Extracts isolated characters from the source image.

@param mathRawImg Source image.
@param srcRect Ractangular boundary of each isolated character.
@param splitedSrc Extracted bit information of each isolated character.

The function makes use of cv::findContours(...) to find out the contour for each isolated character.
According to the character's outter contour, the character can be cropped out from the source image and
its rectangular boundary can be determined.

@note The elemenrs in the srcRect and the splitedSrc are following the order which is from left to right
considering each isolated character's least X-axis value.
*/
void mathImgSplit(const Mat &srcImg, vector<Rect> &srcRect, vector<Mat> &splitedSrc);

/** @brief Predicts characters' correspoding lables and pads the source characters
to the data set's standard.

@param srcRect Rectangular boundaries of the characters.
@param splitedSrc Extracted bit information of the characters.
@param results Predicted lable for each character.
@param trainImgs Standard images loaded from the data set.
@param trainLables standard images corresponding lables loaded from the data set.
@param rowsNum Standard image's height.
@param colsNum Standard image's width.

The function shrinks or enlarges and makes borders to the extracted characters regarding the image size
in the data set and its' own height-width ratio. The lables of the characters are found according to the
data set using the KNN method.
 */
void mathFindLables(const vector<Rect> &srcRect, vector<Mat> &splitedSrc, vector<int> &results,
					const Mat &trainImgs, const Mat &trainLables, int rowsNum, int colsNum);

/** @brief Direction parameter for the findAll() function.
 */
enum FindDirection
{
	TOP,
	BOTTOM
};

/** @breif Finds out all the characters' index in the srcRect sequence in sprcific direction of the current character.

@param pos Index of the current character for reference in the IDrange.
@param foundID Index of the found character in srcRect or lables within the IDrange.
@param srcRect Rectangular boundaries of the characters.
@param lables Predicted lables of the characters.
@param IDrange Current target elements' index in the srcRect or lables.
@param targetDirection Top or bottom characters to be found.
 */
bool findAll(int pos, vector<int> &foundID, const vector<Rect> &srcRect, const vector<int> &lables, vector<int> &IDrange, FindDirection targetDirection);

/** @brief Finds the LaTeX expression according to the lable.

@param lable Character's predicted lable.

@return The LaTeX expression string of the lable.
 */
string correspondSymbol(int lable);

/** @brief Reconstructs the LaTeX expression within the IDrange.

@param mathExpr Previously generated part of expression.
@param srcRect Rectangular boundaries of the characters.
@param lables Predicted lables of the characters.
@param IDrange Current target elements' index in the srcRect or lables.

The function establishes various cases based upon different characters to reorganize and
reconstruct the LaTeX expression string within the IDrange of the characters, which is usually a block
with a single stucture, like fraction. It uses recursive method to deal with sturcture blocks inside outter
stucture. mathExpr will be modified and finally becomes a completed expression within IDrange
and its previous expression.
 */
bool mathString(string &mathExpr, const vector<Rect> &srcRect, const vector<int> &lables, vector<int> &IDrange);

/** @brief Recontructs the LateX expression of a fraction expression.

@param pos Index in the IDrange of the current character for reference.
@param srcRect Rectangular boundaries of the characters.
@param lables Predicted lables of the characters.
@param IDrange Current target elements' index in the srcRect or lables.

The function takes the franction line as reference to find out the characters' indices of
the numerator and the denominator within the IDrange. Then mathString(...) is called to reconstruct
the LaTeX expression respectively. pos will be modified according to the number of characters
found in the numerator and denominator.

@return The LaTeX expression string of the fraction.
 */
string fractionSort(int &pos, const vector<Rect> &srcRect, const vector<int> &lables, vector<int> &IDrange);

/** @brief Recontructs the LaTeX expression of a character' superscript and subscript.

@param pos Index in the IDrange of the current character for reference.
@param srcRect Rectangular boundaries of the characters.
@param lables Predicted lables of the characters.
@param IDrange Current target elements' index in the srcRect or lables.

The function takes current character as reference to find out the characters' indices of
the superscript and the subscript within the IDrange. Then mathString(...) is called to reconstruct
the LaTeX expression respectively. pos will be modified according to the number of characters found in
the superscript and subscript.

@return The LaTeX expression string of the superscripts and subscripts.
 */
string scriptSort(int &pos, const vector<Rect> &srcRect, const vector<int> &lables, vector<int> &IDrange);

/** @brief Recontructs the LaTeX expression of a square root expression.

@param pos Index in the IDrange of the current character for reference.
@param srcRect Rectangular boundaries of the characters.
@param lables Predicted lables of the characters.
@param IDrange Current target elements' index in the srcRect or lables.

The function takes the radical symbol as reference to find out the characters' indices of
the square root expression within the IDrange. The mathString(...) is called to reconstruct
the LaTeX expression respectively. pos will be modified according to the number of characters found
in the square root expression.

@return The LaTeX expression string of the square root expression.
 */
string sqrtSort(int &pos, const vector<Rect> &srcRect, const vector<int> &lables, vector<int> &IDrange);

int main()
{
	string YorN;
	while (true)
	{
		cout << "Process standard images of a math font to establish a dataset? (y/N): ";
		cin >> YorN;

		if ('Y' == YorN[0] || 'y' == YorN[0])
		{
			processDataSet();
			break;
		}
		else if ('N' == YorN[0] || 'n' == YorN[0])
		{
			break;
		}
		else
		{
			cout << "Illegal instruction. y(yes) or N(no) only." << endl;
		}
	}

	Mat imgsData;
	Mat lablesData;
	int colSize;
	int rowSize;
	loadDataSet(imgsData, lablesData, rowSize, colSize);

	while (true)
	{
		Mat mathRawImg;
		string rawImgName;
		while (true)
		{
			cout << "Absolute Path to the image to be recognized: " << endl;
			cin >> rawImgName;
			mathRawImg = imread(rawImgName, IMREAD_GRAYSCALE);

			if (mathRawImg.empty())
			{
				cout << "Failed to open the image. Maybe the path is incorrect or the file is not a image file in specific formats." << endl;
			}
			else
			{
				break;
			}
		}

		mathImgPreprocess(mathRawImg);

		vector<Rect> srcRect;
		vector<Mat> splitedSrc;
		mathImgSplit(mathRawImg, srcRect, splitedSrc);

		vector<int> results;
		mathFindLables(srcRect, splitedSrc, results, imgsData, lablesData, rowSize, colSize);

		string outputExpr;

		// Initialize a IDrange that contains all the index in the results in order
		vector<int> IDrange(results.size());
		for (int i = 0; i < IDrange.size(); i++)
		{
			IDrange[i] = i;
		}

		mathString(outputExpr, srcRect, results, IDrange);

		cout << "LaTeX expression result: " << endl;
		cout << outputExpr << endl;

		// Show the processed image to examine if it's decently preprocessed
		namedWindow("Preprocessed Image", WINDOW_KEEPRATIO);
		imshow("Preprocessed Image", mathRawImg);
		waitKey(0);
	}

	return 0;
}

void processDataSet()
{
	cout << "Refer to the README file about the process to establish a set of standard images for each glyph." << endl;

	ofstream printData;
	ofstream printLable;

	string datasetName;
	string lablesetName;

	while (true)
	{
		cout << "Absolute path to the new image data set. (File name is required, suffix is not required. If the file already exists, it will be OVERWRITTEN.): " << endl;
		cin >> datasetName;
		printData.open(datasetName, ios::binary);

		if (printData)
		{
			break;
		}
		else
		{
			cout << "Failed to create or open such a file. Consider checking the path's correctness." << endl;
		}
	}

	while (true)
	{
		cout << "Absolute path to the new lable data set. (File name is required, suffix is not required. If the file already exists, it will be OVERWRITTEN.): " << endl;
		cin >> lablesetName;
		printLable.open(lablesetName, ios::binary);

		if (printLable)
		{
			break;
		}
		else
		{
			cout << "Failed to create or open such a file. Consider checking the path's correctness." << endl;
		}
	}

	int sampleNum;

	string prefix;
	string suffix;
	string num;
	string ImgName;

	while (true)
	{
		cout << "Absolute path to the Folder storing the standard images: " << endl;
		cin >> prefix;
		cout << "Suffix of the image file (Case sensitive, dot(.) should NOT be entered): " << endl;
		cin >> suffix;
		cout << "Number of images in the set: " << endl;
		cin >> sampleNum;

		const int rowSize = 100;
		const int colSize = 100;

		// Magic number is used to indentify the data set
		const int imageMagicNum = 19961111;
		const int lableMagicNum = 19961221;

		printData.write((char *)&imageMagicNum, sizeof(imageMagicNum));
		printData.write((char *)&rowSize, sizeof(rowSize));
		printData.write((char *)&colSize, sizeof(colSize));
		printData.write((char *)&sampleNum, sizeof(sampleNum));

		printLable.write((char *)&lableMagicNum, sizeof(lableMagicNum));
		printLable.write((char *)&sampleNum, sizeof(sampleNum));

		bool normallyProcessed = true;

		for (int i = 0; i < sampleNum; i++)
		{
			stringstream temp;
			temp << i;
			temp >> num;
			ImgName = prefix + "\\" + num + "." + suffix;

			Mat symbolImg;
			symbolImg = imread(ImgName, IMREAD_GRAYSCALE);
			if (symbolImg.empty())
			{
				normallyProcessed = false;
				break;
			}

			symbolImg = ~symbolImg;

			vector<vector<Point>> contour;
			findContours(symbolImg, contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());

			// If the standard image possesses two or more separated symbols then it's not suitable
			if (contour.size() > 1)
			{
				cout << "The NO." << i << " image has two or more contours. Consider checking its validity and regenerate the data set." << endl;
			}

			const int TOTAL_SIZE = 100;
			const int ACTUAL_SIZE = TOTAL_SIZE - TOTAL_SIZE / 10;

			Rect boundary;
			boundary = boundingRect(contour[0]);

			Size imgSize;
			int top, bottom, left, right;
			if (boundary.width > boundary.height)
			{
				imgSize.width = ACTUAL_SIZE;
				imgSize.height = ACTUAL_SIZE * boundary.height / boundary.width;

				left = (TOTAL_SIZE - ACTUAL_SIZE) / 2;
				right = left;
				top = (TOTAL_SIZE - imgSize.height) / 2;
				bottom = TOTAL_SIZE - top - imgSize.height;
			}
			else
			{
				imgSize.height = ACTUAL_SIZE;
				imgSize.width = ACTUAL_SIZE * boundary.width / boundary.height;

				top = (TOTAL_SIZE - ACTUAL_SIZE) / 2;
				bottom = top;
				left = (TOTAL_SIZE - imgSize.width) / 2;
				right = TOTAL_SIZE - left - imgSize.width;
			}

			resize(symbolImg(boundary), symbolImg, imgSize);
			copyMakeBorder(symbolImg, symbolImg, top, bottom, left, right, BORDER_CONSTANT, Scalar::all(0));

			printData.write((char *)symbolImg.data, rowSize * colSize);
			unsigned char lable = i;
			printLable.write((char *)&lable, sizeof(lable));
		} // i < sampleNum

		if (normallyProcessed)
		{
			break;
		}
		else
		{
			printData.close();
			printLable.close();
			cout << "Error occurs. Consider checking the validity path to the folder or the image file names' suffix or the total number of the images." << endl;
			printData.open(datasetName, ios::binary);
			printLable.open(lablesetName, ios::binary);
		}
	} // while(true)

	printData.close();
	printLable.close();
	cout << "Image data set has been established and stored as ' " << datasetName << " '." << endl;
	cout << "Lable data set has been established and stored as ' " << lablesetName << " '." << endl;
	return;
}

void loadDataSet(Mat &trainImgs, Mat &trainLables, int &rowsNum, int &colsNum)
{
	string dataPath;
	string lablePath;
	ifstream trainDataImages;
	ifstream trainDataLables;

	while (true)
	{
		cout << "Absolute path to the standard image data set: " << endl;
		cin >> dataPath;
		trainDataImages.open(dataPath, ios::binary);
		if (!trainDataImages)
		{
			cout << "Could not open the image dataset. Consider checking the path's validity." << endl;
			trainDataImages.close();
		}
		else
		{
			break;
		}
	}

	int imageMagicNum;
	const int IMAGE_MAGIC_NUM = 19961111;
	while (true)
	{
		trainDataImages.read((char *)&imageMagicNum, sizeof(imageMagicNum));

		if (19961111 == imageMagicNum)
		{
			break;
		}
		else
		{
			trainDataImages.close();

			cout << "This file is not a valid image data set. Consider checking the path's correctness." << endl;

			cout << "Absolute path to the standard image data set: " << endl;
			cin >> dataPath;
			trainDataImages.open(dataPath, ios::binary);
		}
	}

	while (true)
	{
		cout << "Absolute path to the lable data set: " << endl;
		cin >> lablePath;
		trainDataLables.open(lablePath, ios::binary);
		if (!trainDataLables)
		{
			cout << "Could not open the lables dataset. Consider checking the path's validity." << endl;
			trainDataLables.close();
		}
		else
		{
			break;
		}
	}

	int lableMagicNum;
	const int LABLE_MAGIC_NUM = 19961221;
	while (true)
	{
		trainDataLables.read((char *)&lableMagicNum, sizeof(lableMagicNum));

		if (LABLE_MAGIC_NUM == lableMagicNum)
		{
			break;
		}
		else
		{
			trainDataLables.close();

			cout << "This file is not a valid lable data set. Consider checking the path's correctness." << endl;

			cout << "Absolute path to the lable data set: " << endl;
			cin >> lablePath;
			trainDataLables.open(lablePath, ios::binary);
		}
	}

	int trainImgsNum;

	trainDataImages.read((char *)&rowsNum, sizeof(rowsNum));
	trainDataImages.read((char *)&colsNum, sizeof(colsNum));
	trainDataImages.read((char *)&trainImgsNum, sizeof(trainImgsNum));

	cout << "The total number of images in the training dataset is: " << trainImgsNum << endl;

	// featureSize is the total pixels of a single standard image
	int featureSize = colsNum * rowsNum;

	trainImgs.create(/*row size*/ trainImgsNum, /*row size*/ featureSize, CV_8UC1);
	trainDataImages.read((char *)trainImgs.data, featureSize * trainImgsNum);

	// Binarize all the pixels of the trainImgs
	threshold(trainImgs, trainImgs, 100, 255, THRESH_BINARY);
	// KNN required
	trainImgs.convertTo(trainImgs, CV_32FC1);

	trainDataImages.close();

	int trainLablesNum;

	trainDataLables.read((char *)&trainLablesNum, sizeof(trainLablesNum));

	trainLables.create(trainLablesNum, 1, CV_8UC1);
	trainDataLables.read((char *)trainLables.data, trainLablesNum);
	trainLables.convertTo(trainLables, CV_32SC1);

	trainDataLables.close();
}

void mathImgPreprocess(Mat &mathRawImg)
{
	blur(mathRawImg, mathRawImg, Size(5, 5));
	threshold(mathRawImg, mathRawImg, /*threshold, useless under Otsu method*/ 100, /*Maximum grayscale value*/ 255, THRESH_OTSU);
	mathRawImg = ~mathRawImg;

	Mat canniedImg;
	Canny(mathRawImg, canniedImg, /*threshold1*/ 50, /*threshold2*/ 200, /*aperture size*/ 3);

	vector<Vec2f> lines;
	// Fisrt time Hough lines transform with a larger interval(CV_PI / 10.0) and a higher threshold
	HoughLines(canniedImg, lines, 1, CV_PI / 10.0, canniedImg.cols / 50);

	vector<int> radianDist1(/*total section*/ 10, 0);
	for (int i = 0; i < lines.size(); i++)
	{
		// Calculate the section which current radian belongs to
		int radian = (int)(/*current line's radian*/ lines[i].val[1] * 10.0 / CV_PI);

		if (radian >= 10)
		{
			continue;
		}

		radianDist1[radian]++;
	}

	int radianMaxNum = 0;
	double radianMostDist1 = 0.0;
	for (int i = 0; i < 10; i++)
	{
		if (radianDist1[i] > radianMaxNum)
		{
			radianMaxNum = radianDist1[i];
			radianMostDist1 = (double)(i)*CV_PI / 10.0;
		}
	}

	lines.clear();
	bool isCLockwiseTilted = (radianMostDist1 > CV_PI / 2);

	// Because the Hough line transform in openCV always produces a value closest to CV_PI/2,
	// there are two different situations choosing the correct section from the first time transform
	if (isCLockwiseTilted)
	{
		HoughLines(canniedImg, lines, 1, CV_PI / 180, canniedImg.cols / 25, 0, 0, radianMostDist1 - CV_PI / 10.0, radianMostDist1);
	}
	else
	{
		HoughLines(canniedImg, lines, 1, CV_PI / 180, canniedImg.cols / 25, 0, 0, radianMostDist1, radianMostDist1 + CV_PI / 10.0);
	}

	vector<int> radianDist2(18, 0);
	for (int i = 0; i < lines.size(); i++)
	{
		int radian;
		if (isCLockwiseTilted)
		{
			radian = (int)((lines[i].val[1] - (radianMostDist1 - CV_PI / 10.0)) * 180.0 / CV_PI);
		}
		else
		{
			radian = (int)((lines[i].val[1] - radianMostDist1) * 180.0 / CV_PI);
		}

		if (radian >= 18)
		{
			continue;
		}

		radianDist2[radian]++;
	}

	radianMaxNum = 0;
	double radianMostDist2 = 0.0;
	for (int i = 0; i < 18; i++)
	{
		if (radianDist2[i] > radianMaxNum)
		{
			radianMaxNum = radianDist2[i];

			if (isCLockwiseTilted)
			{
				radianMostDist2 = (radianMostDist1 - CV_PI / 10.0) + (double)(i)*CV_PI / 180.0;
			}
			else
			{
				radianMostDist2 = radianMostDist1 + (double)(i)*CV_PI / 180.0;
			}
		}
	}

	double rotateRadian = radianMostDist2 - CV_PI / 2.0;
	if (!(abs(rotateRadian - 0.0) < FLT_EPSILON))
	{
		int verticalBorder, horizontalBorder;

		// Because openCV doesn't automatically modify the size of the rotated image,
		// dimensional expansion according to the rotated angle is required to retain
		// all the information of the original image if the scale is pinned down
		if (rotateRadian > 0)
		{
			verticalBorder = ((double)mathRawImg.rows * cos(rotateRadian) + (double)mathRawImg.cols * sin(rotateRadian) - (double)mathRawImg.rows) / 2;
			horizontalBorder = ((double)mathRawImg.cols * cos(rotateRadian) + (double)mathRawImg.rows * sin(rotateRadian) - (double)mathRawImg.cols) / 2;
		}
		else
		{
			verticalBorder = ((double)mathRawImg.rows * cos(rotateRadian) + (double)mathRawImg.cols * sin(0.0 - rotateRadian) - (double)mathRawImg.rows) / 2;
			horizontalBorder = ((double)mathRawImg.cols * cos(rotateRadian) + (double)mathRawImg.rows * sin(0.0 - rotateRadian) - (double)mathRawImg.cols) / 2;
		}

		if (verticalBorder < 0)
			verticalBorder = 0;
		if (horizontalBorder < 0)
			horizontalBorder = 0;
		copyMakeBorder(mathRawImg, mathRawImg, verticalBorder, verticalBorder, horizontalBorder, horizontalBorder, BORDER_CONSTANT, Scalar::all(0));

		Mat rotMat;
		Point center(mathRawImg.cols / 2, mathRawImg.rows / 2);
		double angle = rotateRadian * 180 / CV_PI;
		double scale = 1.0;

		rotMat = getRotationMatrix2D(center, angle, scale);
		warpAffine(mathRawImg, mathRawImg, rotMat, mathRawImg.size());
	}

	morphologyEx(mathRawImg, mathRawImg, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
	morphologyEx(mathRawImg, mathRawImg, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));

	return;
}

void mathImgSplit(const Mat &srcImg, vector<Rect> &srcRect, vector<Mat> &splitedSrc)
{
	Mat mathRawImg;
	srcImg.copyTo(mathRawImg);

	// Because the direction of openCV finding the contours is from bottom to top based on each separated
	// shape's smallest Y-axis value(the highest point). In order to save the process of reordering the contours,
	// the source image is transposed to produce the contours in expected order
	mathRawImg = mathRawImg.t();
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(mathRawImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());

	for (int i = 0; i < contours.size(); i++)
	{
		// After transposed the contours' order is actually form right to left horizontally,
		// they should be processed in reversed order
		srcRect.push_back(boundingRect(contours[contours.size() - 1 - i]));
		if (srcRect.back().area() < mathRawImg.total() / 10000)
		{
			srcRect.pop_back();
			continue;
		}

		// Use each character's outter contours to crop it out from its the original image
		// instead of using the rectangular boundary to avoid characters' overlapping in a dimension
		Mat temp(mathRawImg.size(), CV_8UC1, Scalar::all(0));
		splitedSrc.push_back(temp);
		drawContours(temp, contours, contours.size() - 1 - i, Scalar::all(1), FILLED, LINE_AA);
		mathRawImg.copyTo(splitedSrc.back(), temp);
		splitedSrc.back() = splitedSrc.back()(srcRect.back()).t();

		// The value of the Rectangular boundary should also be exchanged
		swap(srcRect.back().width, srcRect.back().height);
		swap(srcRect.back().x, srcRect.back().y);
	}

	cout << "Total amount of the isolated characters is " << splitedSrc.size() << endl;
	return;
}

void mathFindLables(const vector<Rect> &srcRect, vector<Mat> &splitedSrc, vector<int> &results,
					const Mat &trainImgs, const Mat &trainLables, int rowsNum, int colsNum)
{
	int borderWidth = rowsNum / 10;
	Mat img2Define;
	for (int i = 0; i < splitedSrc.size(); i++)
	{
		Size imgSize;
		int top, bottom, left, right;

		if (srcRect[i].width > srcRect[i].height)
		{
			imgSize.width = rowsNum - borderWidth;

			// In case of some extreme width-height ratio
			if (imgSize.width * srcRect[i].height / srcRect[i].width < /*The height of '-' in the standard image*/ 6)
				imgSize.height = 6;
			else
				imgSize.height = imgSize.width * srcRect[i].height / srcRect[i].width;

			left = borderWidth / 2;
			right = borderWidth - left;
			top = (colsNum - imgSize.height) / 2;
			bottom = colsNum - top - imgSize.height;
		}
		else
		{
			imgSize.height = colsNum - borderWidth;

			// In case of some extreme height-width ratio
			if (imgSize.height * srcRect[i].width / srcRect[i].height < 6)
				imgSize.width = 6;
			else
				imgSize.width = imgSize.height * srcRect[i].width / srcRect[i].height;

			top = borderWidth / 2;
			bottom = borderWidth - top;
			left = (rowsNum - imgSize.width) / 2;
			right = rowsNum - left - imgSize.width;
		}

		resize(splitedSrc[i], splitedSrc[i], imgSize, 0, 0, INTER_NEAREST);
		copyMakeBorder(splitedSrc[i], splitedSrc[i], top, bottom, left, right, BORDER_CONSTANT, Scalar::all(0));

		img2Define.push_back(splitedSrc[i].reshape(0, 1));
	} // i < splitedSrc.size()

	img2Define.convertTo(img2Define, CV_32FC1);

	Ptr<ml::KNearest> knn = ml::KNearest::create();
	knn->train(trainImgs, ml::ROW_SAMPLE, trainLables);

	vector<float> resultsF;
	knn->findNearest(img2Define, 1, resultsF);

	results.assign(resultsF.begin(), resultsF.end());

	return;
}

bool mathString(string &mathExpr, const vector<Rect> &srcRect, const vector<int> &lables, vector<int> &IDrange)
{
	// Record the current relative position of "|"
	bool leftOrRight = 1;

	for (int i = 0; i < IDrange.size(); i++)
	{
		vector<int> topIDs;
		vector<int> bottomIDs;

		int currentPos = i;

		switch (lables[IDrange[i]])
		{
			// First to find the separated dot of 'i' or 'j'
			// If the next character is not a dot, the it is probably the hat
		case 18:
		case 19:
		case 126:
			if (i + 1 >= IDrange.size())
			{
				mathExpr = mathExpr + correspondSymbol(lables[IDrange[i]]);
				cout << "Result may be inaccurate about the NO." << IDrange[i] << " isolated character asembling 'j' or 'i'." << endl;
				break;
			}

			if (99 == lables[IDrange[i + 1]])
			{
				mathExpr = mathExpr + correspondSymbol(lables[IDrange[i]]);
				swap(IDrange[i + 1], IDrange[i]);
				i++;
				mathExpr = mathExpr + scriptSort(i, srcRect, lables, IDrange);
			}
			else if (srcRect[IDrange[i + 1]].x < (srcRect[IDrange[i]].x + srcRect[IDrange[i]].width / 2))
			{
				swap(IDrange[i], IDrange[i + 1]);
				i--;
			}
			else
			{
				mathExpr = mathExpr + correspondSymbol(lables[IDrange[i]]);
				mathExpr = mathExpr + scriptSort(i, srcRect, lables, IDrange);
			}
			break;

			// The sum or the product integral characters
		case 73:
		case 76:
			mathExpr = mathExpr + correspondSymbol(lables[IDrange[i]]);

			findAll(i, topIDs, srcRect, lables, IDrange, TOP);
			if (0 != topIDs.size())
			{
				mathExpr = mathExpr + "^{";
				mathString(mathExpr, srcRect, lables, topIDs);
				mathExpr = mathExpr + "}";
			}

			findAll(i, bottomIDs, srcRect, lables, IDrange, BOTTOM);
			if (0 != bottomIDs.size())
			{
				mathExpr = mathExpr + "_{";
				mathString(mathExpr, srcRect, lables, bottomIDs);
				mathExpr = mathExpr + "}";
			}

			i = i + topIDs.size() + bottomIDs.size();
			break;

			// '-'
			// Four situation of '-': '-', part of '=', a fraction line, a sbar
		case 85:
			if (i + 1 >= IDrange.size())
			{
				mathExpr = mathExpr + "-";
				cout << "Result may be inaccurate about the NO." << IDrange[i] << " isolated character asembling '-' and its following characters." << endl;
			}
			else if (srcRect[IDrange[i + 1]].x > (srcRect[IDrange[i]].x + srcRect[IDrange[i]].width / 2))
				mathExpr = mathExpr + "-";
			else
			{
				if (85 == lables[IDrange[i + 1]])
				{
					if (i + 2 >= IDrange.size())
					{
						mathExpr = mathExpr + '=';
						i++;
						cout << "Result may be inaccurate with the NO." << IDrange[i] << " isolated character asembling '=' and its following characters." << endl;
						break;
					}
					else if ((srcRect[IDrange[i + 2]].x > (srcRect[IDrange[i]].x + srcRect[IDrange[i]].width / 2)))
					{
						mathExpr = mathExpr + "=";
						i++;
						break;
					}
				}

				bool isTopFound = false;
				// Find out if there is any character under the current character
				for (int j = i + 1; j < IDrange.size(); j++)
				{
					if (srcRect[IDrange[j]].x > (srcRect[IDrange[i]].x + srcRect[IDrange[i]].width / 2))
						break;
					else if (srcRect[IDrange[j]].y < (srcRect[IDrange[i]].y + srcRect[IDrange[i]].height))
					{
						isTopFound = true;
						break;
					}
				}

				if (!isTopFound)
				{
					findAll(i, bottomIDs, srcRect, lables, IDrange, BOTTOM);
					if (1 == bottomIDs.size())
					{
						mathExpr = mathExpr + "\\bar{";
						mathExpr = mathExpr + correspondSymbol(lables[IDrange[i + 1]]);
					}
					else if (bottomIDs.size() > 1)
					{
						mathExpr = mathExpr + "\\overline{";
						mathString(mathExpr, srcRect, lables, bottomIDs);
					}
					mathExpr = mathExpr + "}";
					i = i + bottomIDs.size();
					mathExpr = mathExpr + scriptSort(i, srcRect, lables, IDrange);
				}
				else
					mathExpr = mathExpr + fractionSort(i, srcRect, lables, IDrange);
			} // else
			break;

			// '^'
		case 100:
			findAll(i, bottomIDs, srcRect, lables, IDrange, BOTTOM);

			if (0 == bottomIDs.size())
			{
				mathExpr = mathExpr + correspondSymbol(lables[IDrange[i]]);
				break;
			}

			if (1 == bottomIDs.size())
			{
				mathExpr = mathExpr + "\\hat{";
				mathExpr = mathExpr + correspondSymbol(lables[IDrange[i + 1]]);
			}
			else
			{
				mathExpr = mathExpr + "\\widehat{";
				mathString(mathExpr, srcRect, lables, bottomIDs);
			}
			mathExpr = mathExpr + "}";

			i = i + bottomIDs.size();
			mathExpr = mathExpr + scriptSort(i, srcRect, lables, IDrange);
			break;

			// '.'
		case 99:
			if (i + 1 >= IDrange.size())
			{
				mathExpr = mathExpr + correspondSymbol(lables[IDrange[i]]);
				break;
			}
			else
			{
				if (srcRect[IDrange[i + 1]].x < (srcRect[IDrange[i]].x + srcRect[IDrange[i]].width / 2))
				{
					mathExpr = mathExpr + "\\dot{";
					mathExpr = mathExpr + correspondSymbol(lables[IDrange[i + 1]]);
					mathExpr = mathExpr + "}";

					i++;
					mathExpr = mathExpr + scriptSort(i, srcRect, lables, IDrange);
				}
				else
				{
					mathExpr = mathExpr + correspondSymbol(lables[IDrange[i]]);
				}
			}
			break;

			// Vector arrow or normal right arrow
		case 105:
		case 108:
			findAll(i, bottomIDs, srcRect, lables, IDrange, BOTTOM);

			if (0 == bottomIDs.size())
			{
				mathExpr = mathExpr + correspondSymbol(lables[IDrange[i]]);
				break;
			}
			else
			{
				if (1 == bottomIDs.size())
				{
					mathExpr = mathExpr + "\\vec{";
					mathExpr = mathExpr + correspondSymbol(lables[IDrange[i + 1]]);
				}
				else
				{
					mathExpr = mathExpr + "\\overrightarrow{";
					mathString(mathExpr, srcRect, lables, bottomIDs);
				}
				mathExpr = mathExpr + "}";
			}

			i = i + bottomIDs.size();
			mathExpr = mathExpr + scriptSort(i, srcRect, lables, IDrange);
			break;

		// Different size of the radical symbol
		case 54:
		case 104:
		case 109:
			mathExpr = mathExpr + sqrtSort(i, srcRect, lables, IDrange);

			if ("" != scriptSort(currentPos, srcRect, lables, IDrange))
			{
				cout << "" << endl;
			}
			break;

			// '|'
		case 116:
			if (1 == leftOrRight)
			{
				mathExpr = mathExpr + "\\left|";
				leftOrRight = 0;
			}
			else
			{
				mathExpr = mathExpr + "\\right|";
				leftOrRight = 1;
			}
			mathExpr = mathExpr + scriptSort(i, srcRect, lables, IDrange);
			break;

			// These are the characters that are not likely to have ant superscript or subscript
		case 50:
		case 61:
		case 84:
		case 86:
		case 87:
		case 88:
		case 95:
		case 98:
		case 110:
		case 111:
		case 112:
		case 113:
		case 114:
		case 115:
		case 138:
		case 139:
		case 140:
		case 142:
		case 169:
			mathExpr = mathExpr + correspondSymbol(lables[IDrange[i]]);
			break;

		case 1:
			// Determine if it's a upright 'i' which body is resemble to '1'
			if (i + 1 < IDrange.size() && 99 == lables[IDrange[i + 1]] &&
				srcRect[IDrange[i + 1]].x < (srcRect[IDrange[i]].x + srcRect[IDrange[i]].width / 2))
			{
				mathExpr = mathExpr + "\\mathrm{i}";
				i++;
				break;
			}

			// Determine if it's the 'l' inside a 'lim'
			findAll(i, bottomIDs, srcRect, lables, IDrange, BOTTOM);
			if (0 != bottomIDs.size())
			{
				int msID = IDrange.size();
				for (int j = i + 1; j < IDrange.size(); j++)
				{
					if (srcRect[IDrange[j]].y > (srcRect[IDrange[i]].y + srcRect[IDrange[i]].height))
						bottomIDs.push_back(IDrange[j]);
					else if (129 == lables[IDrange[j]] && srcRect[IDrange[j]].y < (srcRect[IDrange[i]].y + srcRect[IDrange[i]].height) && IDrange.size() == msID)
						msID = j;
					else if (j > msID && srcRect[IDrange[j]].y < (srcRect[IDrange[i]].y + srcRect[IDrange[i]].height))
						break;
				}

				if (IDrange.size() == msID)
					break;

				mathExpr = mathExpr + "\\lim_{";
				mathString(mathExpr, srcRect, lables, bottomIDs);
				mathExpr = mathExpr + "}";
				i = i + bottomIDs.size() + 3;
				break;
			}

		default:
			// Find out if there is any hat above the current character
			bool isTopOrBottom = false;
			for (int j = i + 1; j < IDrange.size(); j++)
			{
				int pos = IDrange[j];

				if (srcRect[pos].x > (srcRect[IDrange[i]].x + srcRect[IDrange[i]].width / 2))
					break;
				else if (73 == lables[pos] || 76 == lables[pos] || (1 == lables[pos]) || 99 == lables[pos] ||
						 85 == lables[pos] || 100 == lables[pos] || 105 == lables[pos] || 108 == lables[pos])
				{
					swap(IDrange[i], IDrange[j]);
					i--;
					isTopOrBottom = true;
					break;
				}
			}

			if (true == isTopOrBottom)
			{
				break;
			}

			string scriptExpr = "";
			if (!(89 == lables[IDrange[i]] || 91 == lables[IDrange[i]] || 93 == lables[IDrange[i]]))
			{
				scriptExpr = scriptSort(i, srcRect, lables, IDrange);
			}

			if ("" == scriptExpr)
			{
				// Determine if it is the upright 'c' inside 'cos'
				if (119 == lables[IDrange[i]] && i + 2 < IDrange.size())
				{
					if (157 == lables[IDrange[i + 1]] && 161 == lables[IDrange[i + 2]])
					{
						mathExpr = mathExpr + "\\cos ";
						i += 2;

						mathExpr = mathExpr + scriptSort(i, srcRect, lables, IDrange);
						break;
					}
				}
				// Determine if it is the upright 's' inside 'sin'
				else if (161 == lables[IDrange[i]] && i + 3 < IDrange.size())
				{
					if ((99 == lables[IDrange[i + 1]] || 151 == lables[IDrange[i + 1]]) && 99 == lables[IDrange[i + 2]] && 130 == lables[IDrange[i + 3]])
					{
						mathExpr = mathExpr + "\\sin ";
						i += 3;

						mathExpr = mathExpr + scriptSort(i, srcRect, lables, IDrange);
						break;
					}
				}
			}

			mathExpr = mathExpr + correspondSymbol(lables[IDrange[currentPos]]);
			mathExpr = mathExpr + scriptExpr;
			break;
		} // switch (lables[IDrange[i]])
	}	 // i < IDrange.size()

	return true;
}

string fractionSort(int &pos, const vector<Rect> &srcRect, const vector<int> &lables, vector<int> &IDrange)
{
	string fracExpr = "\\frac{";

	vector<int> topIDs;
	findAll(pos, topIDs, srcRect, lables, IDrange, TOP);
	mathString(fracExpr, srcRect, lables, topIDs);

	fracExpr = fracExpr + "}{";

	vector<int> bottomIDs;
	findAll(pos, bottomIDs, srcRect, lables, IDrange, BOTTOM);
	mathString(fracExpr, srcRect, lables, bottomIDs);

	pos = pos + topIDs.size() + bottomIDs.size();

	fracExpr = fracExpr + "}";
	return fracExpr;
}

string scriptSort(int &pos, const vector<Rect> &srcRect, const vector<int> &lables, vector<int> &IDrange)
{
	string scriptExpr = "";

	vector<int> supScriptIDs;
	vector<int> subScriptIDs;

	int topLine = srcRect[IDrange[pos]].y + srcRect[IDrange[pos]].height / 3;
	int bottomLine = srcRect[IDrange[pos]].y + srcRect[IDrange[pos]].height * 2 / 3;
	int middleLine = srcRect[IDrange[pos]].y + srcRect[IDrange[pos]].height / 2;

	int parentLable = lables[IDrange[pos]];

	for (int i = pos + 1; i < IDrange.size(); i++)
	{
		int thisID = IDrange[i];
		int nextID;
		int afterID;

		if (i + 1 < IDrange.size())
		{
			nextID = IDrange[i + 1];
		}
		else
		{
			nextID = -1;
		}

		if (i + 2 < IDrange.size())
		{
			afterID = IDrange[i + 2];
		}
		else
		{
			afterID = -1;
		}

		// Characters like ',' and ''' may be confused with superscript or subscript
		if ((pos + 1) == i && (98 == lables[thisID] || 115 == lables[thisID]))
			break;

		// Following is based on the fact that ',' can not be the last character of superscript or subscript
		else if (115 == lables[thisID])
		{
			if (-1 == nextID)
			{
				cout << "Result may be inaccurate about the NO." << thisID << " character asembling ',' and its following characters." << endl;
				break;
			}

			else
			{
				// normal non-sub/sup script '-' after ','
				if (85 == lables[nextID] && (srcRect[nextID].y + srcRect[nextID].height) > topLine && srcRect[nextID].y < bottomLine)
				{
					break;
				}

				// Normal characters after ','
				if (90 == parentLable || 92 == parentLable || 94 == parentLable || 97 == parentLable || 116 == parentLable)
				{
					if ((srcRect[nextID].y + srcRect[nextID].height) > middleLine && srcRect[nextID].y < middleLine)
					{
						break;
					}
				}
				else
				{
					if ((srcRect[nextID].y + srcRect[nextID].height) > bottomLine && srcRect[nextID].y < middleLine)
					{
						break;
					}
				}

				// Hats after ','
				if (85 == lables[nextID] || 100 == lables[nextID] || 105 == lables[nextID] || 108 == lables[nextID] || 99 == lables[nextID])
				{
					if (-1 == afterID)
					{
						cout << "Result may be inaccurate about the NO." << thisID << " character asembling ',' and its following characters." << endl;
						break;
					}

					else
					{
						vector<int> bottomIDs;
						findAll(i + 1, bottomIDs, srcRect, lables, IDrange, BOTTOM);

						bool isTop = false;
						for (int j = 0; j < bottomIDs.size(); j++)
						{
							if (90 == parentLable || 92 == parentLable || 94 == parentLable || 97 == parentLable || 116 == parentLable)
							{
								if ((srcRect[bottomIDs[j]].y + srcRect[bottomIDs[j]].height) > middleLine && srcRect[bottomIDs[j]].y < middleLine)
								{
									isTop = true;
									break;
								}
							}
							else
							{
								if ((srcRect[bottomIDs[j]].y + srcRect[bottomIDs[j]].height) > bottomLine && srcRect[bottomIDs[j]].y < topLine)
								{
									isTop = true;
									break;
								}
							}
						}

						if (true == isTop)
						{
							break;
						}
					} // if (-1 = afterID)' else
				}	 // if (85 == lables[nextID] || 100 == lables[nextID] || 105 == lables[nextID] || 108 == lables[nextID] || 99 == lables[nextID])
			}		  // if (-1 == nextID)' else
		}			  // if (115 == lables[thisID])

		// '-'
		if (85 == lables[thisID])
		{
			// normal non-sub/sup script '-'
			if ((srcRect[thisID].y + srcRect[thisID].height) > topLine && srcRect[thisID].y < bottomLine)
			{
				break;
			}

			if (-1 == nextID)
			{
				cout << "Result may be inaccurate about the NO." << thisID << " isolated character resembling '-' and its following characters." << endl;
				break;
			}
			// non-sub/sup script '='
			else
			{
				if (85 == lables[nextID] && srcRect[nextID].x < (srcRect[thisID].x + srcRect[thisID].width / 2))
				{
					if (-1 == afterID)
					{
						cout << "Result may be inaccurate about the NO." << thisID << " isolated character resembling '-' or '=' and its following characters." << endl;
						break;
					}
					// Next character of non-sub/sup script '=' must not overlapped vertically
					else if (srcRect[afterID].x > (srcRect[thisID].x + srcRect[thisID].width / 2))
					{
						// if the character is not '=' in the sub/sup script
						if (!((srcRect[thisID].y > middleLine && srcRect[nextID].y > middleLine) ||
							  ((srcRect[thisID].y + srcRect[thisID].height) < middleLine && (srcRect[nextID].y + srcRect[nextID].height) < middleLine)))
						{
							// if the characters are not '-' in the sub and sup script separately
							if (!((srcRect[thisID].y > bottomLine || srcRect[nextID].y > bottomLine) &&
								  ((srcRect[thisID].y + srcRect[thisID].height) < topLine || (srcRect[nextID].y + srcRect[nextID].height) < topLine)))
							{
								break;
							}
						}
					}
				}
			}
		} // if (85 == lables[thisID])

		// '.' and '*'
		if (99 == lables[thisID] || 88 == lables[thisID])
		{
			if (90 == parentLable || 92 == parentLable || 94 == parentLable || 97 == parentLable || 116 == parentLable)
			{
				if ((srcRect[thisID].y + srcRect[thisID].height) < bottomLine && srcRect[thisID].y > topLine)
				{
					break;
				}
			}
			else
			{
				if ((srcRect[thisID].y + srcRect[thisID].height) > middleLine && srcRect[thisID].y < middleLine)
				{
					break;
				}
			}
		}

		// hats and bars...
		if (85 == lables[thisID] || 100 == lables[thisID] || 105 == lables[thisID] || 108 == lables[thisID] || 99 == lables[thisID])
		{
			if (-1 == nextID)
			{
				cout << "Result may be inaccurate about the NO." << thisID << " isolated character resembling '-' or '^' or '.' or arrows and its following characters." << endl;
				break;
			}

			else
			{
				vector<int> bottomIDs;
				findAll(i, bottomIDs, srcRect, lables, IDrange, BOTTOM);

				bool isTop = false;
				for (int j = 0; j < bottomIDs.size(); j++)
				{
					if (90 == parentLable || 92 == parentLable || 94 == parentLable || 97 == parentLable || 116 == parentLable)
					{
						if ((srcRect[bottomIDs[j]].y + srcRect[bottomIDs[j]].height) > middleLine && srcRect[bottomIDs[j]].y < middleLine)
						{
							isTop = true;
							break;
						}
					}
					else
					{
						if ((srcRect[bottomIDs[j]].y + srcRect[bottomIDs[j]].height) > bottomLine && srcRect[bottomIDs[j]].y < topLine)
						{
							isTop = true;
							break;
						}
					}
				}

				if (true == isTop)
				{
					break;
				}
			}
		} // if (85 == lables[thisID] || 100 == lables[thisID] || 105 == lables[thisID] || 108 == lables[thisID] || 99 == lables[thisID])

		if (90 == parentLable || 92 == parentLable || 94 == parentLable || 97 == parentLable || 116 == parentLable)
		{
			if ((srcRect[thisID].y + srcRect[thisID].height) < middleLine)
				supScriptIDs.push_back(thisID);
			else if (srcRect[thisID].y > middleLine)
				subScriptIDs.push_back(thisID);
			else
				break;
		}
		else
		{
			if ((srcRect[thisID].y + srcRect[thisID].height) < bottomLine)
				supScriptIDs.push_back(thisID);
			else if (srcRect[thisID].y > middleLine)
				subScriptIDs.push_back(thisID);
			else
				break;
		}
	} // i < IDrange.size()

	if (0 != subScriptIDs.size())
	{
		scriptExpr = scriptExpr + "_{";
		mathString(scriptExpr, srcRect, lables, subScriptIDs);
		scriptExpr = scriptExpr + "}";
	}
	if (0 != supScriptIDs.size())
	{
		scriptExpr = scriptExpr + "^{";
		mathString(scriptExpr, srcRect, lables, supScriptIDs);
		scriptExpr = scriptExpr + "}";
	}

	pos = pos + supScriptIDs.size() + subScriptIDs.size();
	return scriptExpr;
}

string sqrtSort(int &pos, const vector<Rect> &srcRect, const vector<int> &lables, vector<int> &IDrange)
{
	string sqrtExpr;
	sqrtExpr = sqrtExpr + "\\sqrt";

	bool sqrtFault;
	vector<int> bottomIDs;
	for (int i = pos + 1; i < IDrange.size(); i++)
	{
		// To maintain the continuance of the characters' indices. Characters that do not belong to a square root expression
		// but also can not be structured will be count as ones under the radical symbol to minimize the penalty from the poorly image preprocess
		if ((srcRect[IDrange[i]].x + srcRect[IDrange[i]].width / 2) < (srcRect[IDrange[pos]].x + srcRect[IDrange[pos]].width) &&
			srcRect[IDrange[i]].y < srcRect[IDrange[pos]].y)
		{
			cout << "Result may be inaccurate about the NO." << IDrange[i] << " isolated character that seems to belong to a square root operation." << endl;
			bottomIDs.push_back(IDrange[i]);
		}
		else if ((srcRect[IDrange[i]].x + srcRect[IDrange[i]].width / 2) < (srcRect[IDrange[pos]].x + srcRect[IDrange[pos]].width))
		{
			bottomIDs.push_back(IDrange[i]);
		}
		else
		{
			break;
		}
	}
	sqrtExpr = sqrtExpr + "{";
	mathString(sqrtExpr, srcRect, lables, bottomIDs);
	sqrtExpr = sqrtExpr + "}";

	if (0 == bottomIDs.size())
	{
		cout << "Result may be inaccurate about the NO." << IDrange[pos] << " isolated character resembling the radical symbol and its following characters." << endl;
	}

	pos = pos + bottomIDs.size();
	return sqrtExpr;
}

bool findAll(int pos, vector<int> &foundID, const vector<Rect> &srcRect, const vector<int> &lables, vector<int> &IDrange, FindDirection targetDirection)
{
	switch (targetDirection)
	{
	case TOP:
		for (int i = pos + 1; i < IDrange.size(); i++)
		{
			if (srcRect[IDrange[i]].x < (srcRect[IDrange[pos]].x + srcRect[IDrange[pos]].width))
			{
				if ((srcRect[IDrange[i]].y + srcRect[IDrange[i]].height) < srcRect[IDrange[pos]].y)
					foundID.push_back(IDrange[i]);
			}
			else
				break;
		}
		break;
	case BOTTOM:
		for (int i = pos + 1; i < IDrange.size(); i++)
		{
			if (srcRect[IDrange[i]].x < (srcRect[IDrange[pos]].x + srcRect[IDrange[pos]].width))
			{
				if (srcRect[IDrange[i]].y > (srcRect[IDrange[pos]].y + srcRect[IDrange[pos]].height))
					foundID.push_back(IDrange[i]);
			}
			else
				break;
		}
		break;
	default:
		break;
	}

	if (0 != foundID.size())
		return true;
	else
		return false;
}

string correspondSymbol(int lable)
{
	if (lable >= 0 && lable <= 9)
	{
		string tmpStr;
		char tmpChar = 48 + lable;
		tmpStr = tmpChar;
		return tmpStr;
	}
	else if (lable >= 10 && lable <= 35)
	{
		string tmpStr;
		char tmpChar = 97 + lable - 10;
		tmpStr = tmpChar;
		return tmpStr;
	}
	// Some italic uppercase letters which are similar to their lowercase ones are ommitted
	else if (lable >= 36 && lable <= 61 &&
			 lable != 50 && lable != 54 && lable != 61)
	{
		string tmpStr;
		char tmpChar = 65 + lable - 36;
		tmpStr = tmpChar;
		return tmpStr;
	}
	// Some upright lowecase letters which are similar to their uppercase ones are ommitted
	else if (lable >= 117 && lable <= 142 &&
			 lable != 125 && lable != 128 && lable != 131 && lable != 135 &&
			 lable != 138 && lable != 139 && lable != 140 && lable != 142)
	{
		string tmpStr;
		char tmpChar = 97 + lable - 117;
		tmpStr = tmpChar;
		tmpStr = "\\mathrm{" + tmpStr + "}";
		return tmpStr;
	}
	else if (lable >= 143 && lable <= 168)
	{
		string tmpStr;
		char tmpChar = 65 + lable - 143;
		tmpStr = tmpChar;
		tmpStr = "\\mathrm{" + tmpStr + "}";
		return tmpStr;
	}
	else
	{
		switch (lable)
		{
		case 50:
			return "\\in ";

		case 61:
			return "\\not\\in ";
		case 62:
			return "\\alpha ";
		case 63:
			return "\\beta ";
		case 64:
			return "\\gamma ";
		case 65:
			return "\\delta ";
		case 66:
			return "\\Delta ";
		case 67:
			return "\\epsilon ";
		case 68:
			return "\\theta ";
		case 69:
			return "\\lambda ";
		case 70:
			return "\\mu ";
		case 71:
			return "\\xi ";
		case 72:
			return "\\pi ";
		case 73:
			return "\\prod ";
		case 74:
			return "\\rho ";
		case 75:
			return "\\sigma ";
		case 76:
			return "\\sum ";
		case 77:
			return "\\tau ";
		case 78:
			return "\\phi ";
		case 79:
			return "\\Phi ";
		case 80:
			return "\\psi ";
		case 81:
			return "\\Psi ";
		case 82:
			return "\\omega ";
		case 83:
			return "\\Omega ";

		case 84:
			return "+";
		case 85:
			return "-";
		case 86:
			return "\\times ";
		case 87:
			return "/";
		case 88:
			return "*";
		case 89:
			return "\\left(";
		case 90:
			return "\\right)";
		case 91:
			return "\\left[";
		case 92:
			return "\\right]";
		case 93:
			return "\\left\\{";
		case 94:
			return "\\right\\}";
		case 95:
			return "<";
		case 96:
			return ">";
		case 97:
			return "\\int ";
		case 98:
			return "'";
		case 99:
			return "\\cdot ";
		case 100:
			return "^";
		case 101:
			return "\\infty ";
		case 102:
			return "\\nabla ";
		case 103:
			return "\\partial ";

		case 106:
			return "\\emptyset ";
		case 107:
			return "\\leftarrow ";
		case 108:
			return "\\rightarrow ";

		case 110:
			return "\\Leftrightarrow ";
		case 111:
			return "\\Rightarrow ";
		case 112:
			return "\\ne ";
		case 113:
			return "\\pm ";
		case 114:
			return "\\propto ";
		case 115:
			return ",";
		case 125:
			return "\\Gamma ";
		case 128:
			return "\\eta ";
		case 131:
			return "\\cup ";
		case 135:
			return "\\hat ";
		case 138:
			return "\\ominus ";
		case 139:
			return "\\oplus ";
		case 140:
			return "\\odot ";
		case 142:
			return "\\otimes ";

		default:
			return "";
		}
	}
}