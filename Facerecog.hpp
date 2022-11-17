//
//  Facerecog.hpp
//  pcafacerecog
//
//  Created by Colton Shepherd on 4/12/22.
//

#ifndef Facerecog_hpp
#define Facerecog_hpp

#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>
#include <vector>

class Facerecog
{
public:
    Facerecog();
    std::vector<cv::Mat> gatherImages(std::string imagestring);
    cv::Mat flattenimages(std::vector<cv::Mat> images);
    cv::SVD getsvd(std::vector<cv::Mat> images);
    cv::Mat framemultiply(cv::Mat frame, cv::SVD svd);
    float imageDistance(cv::Mat firstim, cv::Mat secondim);
    int recognizefacedist(std::vector<cv::Mat> trainImages, cv::Mat testIm,cv::SVD svd, std::vector<int> labels);
    void eyeswapfilter(cv::Mat face);
    ~Facerecog();
private:
    cv::Mat flatimage;
    cv::Mat meanarray;
    
    cv::PCA pca;
    std::vector<cv::Mat> flatimages;
    cv::Mat meanmat;
    cv::Mat dotTransposeMatrix;
    cv::Mat eigenvalsmat;
    cv::Mat eigenvecsmat;
};


#endif /* Facerecog_hpp */
