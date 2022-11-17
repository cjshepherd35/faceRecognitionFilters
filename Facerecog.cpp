//
//  Facerecog.cpp
//  pcafacerecog
//
//  Created by Colton Shepherd on 4/12/22.
//

#include "Facerecog.hpp"


Facerecog::Facerecog()
{
    
}

std::vector<cv::Mat> Facerecog::gatherImages(std::string imagestring)
{
    std::vector<cv::Mat> trainingimages;
    std::vector<cv::String> fn;
    std::string filename = imagestring;
    
    cv::glob(filename, fn, false);
    
    size_t count = fn.size();
        
    for (size_t i = 0; i < count; i++)
    {
        if (!fn[i].empty()) {
            trainingimages.push_back(cv::imread(fn[i], cv::IMREAD_GRAYSCALE));
            
        } else {std::cout << "fn is empty.";}
    }
//    trainingimages.erase(trainingimages.begin());
    for (int i = 0; i < trainingimages.size(); i++) {
        trainingimages[i] = trainingimages[i].reshape(1, 1);
        trainingimages[i].convertTo(trainingimages[i], CV_64F);
        trainingimages[i] = trainingimages[i] / 255.f;
    }
    return trainingimages;
}


cv::Mat Facerecog::flattenimages(std::vector<cv::Mat> images)
{
//    cv::Mat flatmat(images[0].total(), images.size(), CV_64F);
    cv::Mat flatmat;
    cv::Mat newflat;
    for (int i = 1; i < images.size(); i++) {
        cv::Mat changeim = images[i].reshape(1,1);
        flatmat.push_back(changeim);
    }
    flatmat.convertTo(newflat, CV_64F);
    return newflat;
}


cv::SVD Facerecog::getsvd(std::vector<cv::Mat> images)
{
    cv::Mat flatmat = flattenimages(images);
    
    cv::SVD svdtrain = cv::SVD(flatmat);
    return svdtrain;
}

cv::Mat Facerecog::framemultiply(cv::Mat frame, cv::SVD svd)
{
    cv::Mat newframe = frame.reshape(1, frame.cols*frame.rows);
    cv::Mat framedoub;
    newframe.convertTo(framedoub, CV_64F);
    cv::Mat dotpro = svd.vt * framedoub;
    return dotpro;
}

float Facerecog::imageDistance(cv::Mat firstim, cv::Mat secondim)
{
    cv::Mat diff = firstim - secondim;
    cv::Mat sqdiff;
    cv::pow(diff, 2, sqdiff);
    float euclid = cv::sum(sqdiff)[0];
    return euclid;
}

int Facerecog::recognizefacedist(std::vector<cv::Mat> trainImages, cv::Mat testIm, cv::SVD svd, std::vector<int> labels)
{
    std::vector<cv::Mat> flatTrain;
    for (int i = 0; i < trainImages.size(); i++) {
        flatTrain.push_back(framemultiply(trainImages[i], svd));
    }
    
    
    int chosenlabel = -1;
    float smallestval = imageDistance(flatTrain[0], testIm);
    if (smallestval < imageDistance(flatTrain[1], testIm)) {
        chosenlabel = labels[0];
    }
    for (int i = 1; i < flatTrain.size(); i++) {
        float newdist = imageDistance(flatTrain[i], testIm);
        if (newdist < smallestval) {
            smallestval = newdist;
            chosenlabel = labels[i];
        }
    }
    
    return chosenlabel;
}


Facerecog::~Facerecog()
{
    
}
