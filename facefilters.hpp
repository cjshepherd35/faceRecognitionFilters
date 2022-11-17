//
//  facefilters.hpp
//  macopencvobjc
//
//  Created by Colton Shepherd on 4/30/22.
//

#ifndef facefilters_hpp
#define facefilters_hpp

#include <stdio.h>
#include <iostream>
//#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>

class Filters
{
public:
    Filters();
    void makeblack(cv::Mat frame, std::vector<cv::Rect> faces);
    cv::Mat fireface(cv::Mat frame, cv::Rect face);
    cv::Mat eyeswapfilter(cv::Mat face_rect);
    cv::Mat cartoonify(cv::Mat face);
    ~Filters();
private:
    int j, x;
};

#endif /* facefilters_hpp */
