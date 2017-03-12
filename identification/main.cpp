// Main.cpp
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>
#include<iostream>
#include <vector>
using namespace cv;
const int MIN_CONTOUR_AREA = 100;
const cv::Size GAUSSIAN_SMOOTH_FILTER_SIZE = cv::Size(5, 5);
const int ADAPTIVE_THRESH_BLOCK_SIZE = 9;
const int ADAPTIVE_THRESH_WEIGHT = 9;
const int RESIZED_CHAR_IMAGE_WIDTH = 20;
const int RESIZED_CHAR_IMAGE_HEIGHT = 30;

using std::vector;
class PossibleChar {
public:
    // member variables ///////////////////////////////////////////////////////////////////////////
    std::vector<cv::Point> contour;

    cv::Rect boundingRect;

    int intCenterX;
    int intCenterY;

    double dblDiagonalSize;
    double dblAspectRatio;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool sortCharsLeftToRight(const PossibleChar &pcLeft, const PossibleChar & pcRight) {
        return(pcLeft.intCenterX < pcRight.intCenterX);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    bool operator == (const PossibleChar& otherPossibleChar) const {
        if (this->contour == otherPossibleChar.contour) return true;
        else return false;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    bool operator != (const PossibleChar& otherPossibleChar) const {
        if (this->contour != otherPossibleChar.contour) return true;
        else return false;
    }

    // function prototypes ////////////////////////////////////////////////////////////////////////
    PossibleChar(std::vector<cv::Point> _contour);

};
std::vector<PossibleChar> findPossibleCharsInSceneBar(cv::Mat &imgThresh) ;
///////////////////////////////////////////////////////////////////////////////////////////////////
cv::Ptr<cv::ml::KNearest> kNearest = cv::ml::KNearest::create();
void preprocess(cv::Mat &imgOriginal, cv::Mat &imgGrayscale, cv::Mat &imgThresh);
std::string recognizeChars(cv::Mat &imgThresh, std::vector<PossibleChar> &vectorOfMatchingChars);

///////////////////////////////////////////////////////////////////////////////////////////////////
bool loadKNNDataAndTrainKNN(void) {

    // read in training classifications ///////////////////////////////////////////////////

    cv::Mat matClassificationInts;              // we will read the classification numbers into this variable as though it is a vector

    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);        // open the classifications file

    if (fsClassifications.isOpened() == false) {                                                        // if the file was not opened successfully
        std::cout << "error, unable to open training classifications file, exiting program\n\n";        // show error message
        return(false);                                                                                  // and exit program
    }

    fsClassifications["classifications"] >> matClassificationInts;          // read classifications section into Mat classifications variable
    fsClassifications.release();                                            // close the classifications file

                                                                            // read in training images ////////////////////////////////////////////////////////////

    cv::Mat matTrainingImagesAsFlattenedFloats;         // we will read multiple images into this single image variable as though it is a vector

    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);              // open the training images file

    if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
        std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
        return(false);                                                                          // and exit program
    }

    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           // read images section into Mat training images variable
    fsTrainingImages.release();                                                 // close the traning images file

                                                                                // train //////////////////////////////////////////////////////////////////////////////

                                                                                // finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
                                                                                // even though in reality they are multiple images / numbers
    kNearest->setDefaultK(1);

    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

    return true;
}
int main(void) {

    bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();           // attempt KNN training

    if (blnKNNTrainingSuccessful == false) {                            // if KNN training was not successful
                                                                        // show error message
        std::cout << std::endl << std::endl << "error: error: KNN traning was not successful" << std::endl << std::endl;
        return(0);                                                      // and exit program
    }

    cv::Mat imgOriginalScene;           // input image

    imgOriginalScene = cv::imread("digits.png");         // open image

    if (imgOriginalScene.empty()) {                             // if unable to open image
        std::cout << "error: image not read from file\n\n";     // show error message on command line
        //_getch();                                               // may have to modify this line if not using Windows
        return(0);                                              // and exit program
    }
    ////////////////////////////////////////////////
    cv::Mat imgThreshScene;
    cv::Mat imgGrayScaleScene;
    cv::Mat imgBlurred;
    cv::Mat imgContours(imgOriginalScene.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));

    std::vector<std::vector<cv::Point> > ptContours;        // declare contours vector
    std::vector<cv::Vec4i> v4iHierarchy;
    cv::cvtColor(imgOriginalScene, imgGrayScaleScene, CV_BGR2GRAY);        // convert to grayscale

    cv::GaussianBlur(imgGrayScaleScene,imgBlurred,cv::Size(5, 5),0);


    cv::adaptiveThreshold(imgBlurred,imgThreshScene, 255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY_INV,11,2);

   // preprocess(imgOriginalScene, imgGrayScaleScene, imgThreshScene);
    cv::findContours(imgThreshScene,ptContours, v4iHierarchy,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);//cv::CHAIN_APPROX_SIMPLE);

    std::vector<PossibleChar> vectorOfPossibleCharsInScene;
    for(int i=0;i<ptContours.size();i++)
    {
        if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA/20)
        {
            cv::Rect boundingRect = cv::boundingRect(ptContours[i]);                // get the bounding rect
            vectorOfPossibleCharsInScene.push_back(PossibleChar(ptContours[i]));
            cv::rectangle(imgOriginalScene, boundingRect, cv::Scalar(0, 255, 0), 2);
        }
    }
    cv::imshow("hheh",imgOriginalScene);
    //cv::waitKey();
     std::sort(vectorOfPossibleCharsInScene.begin(),vectorOfPossibleCharsInScene.end(),[](PossibleChar char1,PossibleChar char2){return char1.intCenterX<char2.intCenterX;});
     std::string strChars=recognizeChars(imgThreshScene,vectorOfPossibleCharsInScene);
     std::cout<<"results is :"<<strChars;

     int iPause;
     std::cin>>iPause;
     cv::waitKey(0);                 // hold windows open until user presses a key

    return(0);
}
cv::Mat extractValue(cv::Mat &imgOriginal) {
    cv::Mat imgHSV;
    std::vector<cv::Mat> vectorOfHSVImages;
    cv::Mat imgValue;

    cv::cvtColor(imgOriginal, imgHSV, CV_BGR2HSV);

    cv::split(imgHSV, vectorOfHSVImages);

    imgValue = vectorOfHSVImages[2];

    return(imgValue);
}
cv::Mat maximizeContrast(cv::Mat &imgGrayscale) {
    cv::Mat imgTopHat;
    cv::Mat imgBlackHat;
    cv::Mat imgGrayscalePlusTopHat;
    cv::Mat imgGrayscalePlusTopHatMinusBlackHat;

    cv::Mat structuringElement = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3, 3));

    cv::morphologyEx(imgGrayscale, imgTopHat, CV_MOP_TOPHAT, structuringElement);
    cv::morphologyEx(imgGrayscale, imgBlackHat, CV_MOP_BLACKHAT, structuringElement);

    imgGrayscalePlusTopHat = imgGrayscale + imgTopHat;
    imgGrayscalePlusTopHatMinusBlackHat = imgGrayscalePlusTopHat - imgBlackHat;

    return(imgGrayscalePlusTopHatMinusBlackHat);
}
void preprocess(cv::Mat &imgOriginal, cv::Mat &imgGrayscale, cv::Mat &imgThresh)
 {
    imgGrayscale = extractValue(imgOriginal);                           // extract value channel only from original image to get imgGrayscale

    cv::Mat imgMaxContrastGrayscale = maximizeContrast(imgGrayscale);       // maximize contrast with top hat and black hat

    cv::Mat imgBlurred;

    cv::GaussianBlur(imgMaxContrastGrayscale, imgBlurred, GAUSSIAN_SMOOTH_FILTER_SIZE, 0);          // gaussian blur

                                                                                                    // call adaptive threshold to get imgThresh
    cv::adaptiveThreshold(imgBlurred, imgThresh, 255.0, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, 10);//adaptiveThreshold
}

std::string recognizeChars(cv::Mat &imgThresh, std::vector<PossibleChar> &vectorOfMatchingChars) {
    std::string strChars;               // this will be the return value, the chars in the lic plate

    cv::Mat imgThreshColor;

    // sort chars from left to right
    std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);

    cv::cvtColor(imgThresh, imgThreshColor, CV_GRAY2BGR);       // make color version of threshold image so we can draw contours in color on it

    for (auto &currentChar : vectorOfMatchingChars) {           // for each char in plate
        cv::rectangle(imgThreshColor, currentChar.boundingRect, cv::Scalar(0,255,0), 2);       // draw green box around the char

        cv::Mat imgROItoBeCloned = imgThresh(currentChar.boundingRect);                 // get ROI image of bounding rect

        cv::Mat imgROI = imgROItoBeCloned.clone();      // clone ROI image so we don't change original when we resize

        cv::Mat imgROIResized;
        // resize image, this is necessary for char recognition
        cv::resize(imgROI, imgROIResized, cv::Size(RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT));

        cv::Mat matROIFloat;

        imgROIResized.convertTo(matROIFloat, CV_32FC1);         // convert Mat to float, necessary for call to findNearest

        cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);       // flatten Matrix into one row

        cv::Mat matCurrentChar(0, 0, CV_32F);                   // declare Mat to read current char into, this is necessary b/c findNearest requires a Mat

        kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     // finally we can call find_nearest !!!

        float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);       // convert current char from Mat to float

        strChars = strChars + char(int(fltCurrentChar));        // append current char to full string
    }



    return(strChars);               // return result
}
PossibleChar::PossibleChar(std::vector<cv::Point> _contour) {
    contour = _contour;

    boundingRect = cv::boundingRect(contour);

    intCenterX = (boundingRect.x + boundingRect.x + boundingRect.width) / 2;
    intCenterY = (boundingRect.y + boundingRect.y + boundingRect.height) / 2;

    dblDiagonalSize = sqrt(pow(boundingRect.width, 2) + pow(boundingRect.height, 2));

    dblAspectRatio = (float)boundingRect.width / (float)boundingRect.height;
}
