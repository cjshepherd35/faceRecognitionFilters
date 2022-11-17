//
//  imagetest.hpp
//  macopencvobjc
//
//  Created by Colton Shepherd on 3/23/22.
//

#ifndef imagetest_hpp
#define imagetest_hpp

#include <stdio.h>
#include <iostream>
//#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
#include "Facerecog.hpp"
//using namespace cv;
//using namespace cv::face;
#endif /* imagetest_hpp */


class MakePicture {
public:
    MakePicture();
    void showPicture();
    void showFace();
    void makeblack(cv::Mat frame, std::vector<cv::Rect> faces);
    void setblack() {isblack = !isblack;}
    void setfiltername(const char* personfilter);
//    void printfiltername() {std::cout << chosenfilter << " is filer now \n";}
    void recognizeFaces();
    void trainface();
    void fireface();
    void testText();

private:
    bool isblack = false;
    std::string topfile;
    std::string filename;
    std::string name;
    const char* chosenfilter;
    std::string thefilt;
    int filenumber = 0;
    int personnumber = 0;
    std::vector<cv::Mat> trainingimages;
    std::vector<int> labels;
    std::vector<std::string> filters;
    int smallestlabel;
    cv::SVD svd;
};
