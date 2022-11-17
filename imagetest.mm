//
//  imagetest.cpp
//  macopencvobjc
//
//  Created by Colton Shepherd on 3/23/22.
//

#include "imagetest.hpp"
#include "facefilters.hpp"
//#include "videofuncs.mm"
#import "ViewController.h"
#include <stdlib.h>
#include <string>
#include <fstream>
#include "Facerecog.hpp"

using namespace cv::face;
cv::CascadeClassifier face_cascade;

MakePicture::MakePicture()
{
    smallestlabel = -1;
}

void MakePicture::fireface()
{
    cv::CascadeClassifier face_cascade;
//    cv::namedWindow("face window");

    NSString *faceCascadePath = [[NSBundle mainBundle] pathForResource:@"haarcascade_frontalface_alt2"
                                                                    ofType:@"xml"];

    const CFIndex CASCADE_NAME_LEN = 2048;
    char *CASCADE_NAME = (char *) malloc(CASCADE_NAME_LEN);
    CFStringGetFileSystemRepresentation( (CFStringRef)faceCascadePath, CASCADE_NAME, CASCADE_NAME_LEN);

    if(!face_cascade.load(CASCADE_NAME))
    {
        std::cout << "face load failed";

    }
    
    
    NSString *firepath = [[NSBundle mainBundle] pathForResource:@"fireanimation"
                                                                    ofType:@"jpg"];
    const CFIndex fire_length = 2048;
    char *fireanimation = (char *) malloc(fire_length);
    CFStringGetFileSystemRepresentation( (CFStringRef)firepath, fireanimation, fire_length);
    cv::Mat fireanimationreel = cv::imread(fireanimation);
    
    std::string Pname= "";
    int j = 0;
    int x = 0;
    cv::VideoCapture capture(0);
    while (true) {
        cv::Mat frame;
        capture >> frame;
        
        cv::Mat frame_gray;
        cv::cvtColor( frame, frame_gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(frame_gray, frame_gray);
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale( frame_gray, faces );
        
        
        for (int i = 0; i < faces.size(); i++) {
            cv::Rect face_i = faces[i];

            //crop the roi from grya image
            cv::Mat face = frame_gray(face_i);

            //resizing the cropped image to suit to database image sizes
            cv::Mat face_resized;
            cv::resize(face, face_resized, cv::Size(128, 128), 0, 0, cv::INTER_CUBIC);
            
            cv::Rect animrect;

            if (j == 4)
            {
                j = 0;
            }

            switch (j)
            {
                case 0:
                    animrect = cv::Rect((j * fireanimationreel.cols/4), 30, fireanimationreel.cols/4, fireanimationreel.rows/3);
                    break;
                case 1:
                    animrect = cv::Rect(j * fireanimationreel.cols/4, 30, fireanimationreel.cols/4, fireanimationreel.rows/3);
                    break;
                case 2:
                    animrect = cv::Rect((j * fireanimationreel.cols/4)+5, 30, fireanimationreel.cols/4, fireanimationreel.rows/3);
                case 3:
                    animrect = cv::Rect((j * fireanimationreel.cols/4)-15, 30, fireanimationreel.cols/4, fireanimationreel.rows/3);
            }
            
            cv::Mat animation = fireanimationreel(animrect);
            cv::Mat animationcopy;
            cv::resize(animation, animationcopy, cv::Size(face_i.size().width, face_i.size().height/2));
            cv::Rect face_i_copy(face_i.tl().x, face_i.tl().y - 100, face_i.width, face_i.height/2);

            cv::Mat animationgray, finalanim;
            cv::cvtColor(animationcopy, animationgray, cv::COLOR_BGR2GRAY);
            cv::threshold(animationgray, animationgray, 150.0, 255, cv::THRESH_TOZERO);
            animationcopy.copyTo(finalanim, animationgray);


            //blend images.
            bitwise_or(frame(face_i_copy), finalanim, frame(face_i_copy));
//                    animationcopy.copyTo(firelocation);
            animation.release();

            
            if (x > 3) {
                j++;
                x = 0;
            }
            x++;
            
        }
        
        
        cv::imshow("face window", frame);
        if (cv::waitKey(0) >= 0) {
            cv::destroyAllWindows();
            return;
        }
    }
}

void MakePicture::setfiltername(const char* personfilter)
{
    chosenfilter = personfilter;
    thefilt = chosenfilter;
}



//is actually the training funtion
void MakePicture::showPicture()
{
    
    filters.push_back(thefilt);
    
    NSString *faceCascadePath = [[NSBundle mainBundle] pathForResource:@"haarcascade_frontalface_alt2"
                                                                    ofType:@"xml"];
    const CFIndex CASCADE_NAME_LEN = 2048;
    char *CASCADE_NAME = (char *) malloc(CASCADE_NAME_LEN);
    CFStringGetFileSystemRepresentation( (CFStringRef)faceCascadePath, CASCADE_NAME, CASCADE_NAME_LEN);

    if(!face_cascade.load(CASCADE_NAME))
    {
        std::cout << "face load failed";
    }
    
    cv::namedWindow("imwindow");
    std::vector<cv::Rect> faces;
    cv::VideoCapture capture(0);
    int numTrainIms = 0;
    int label = 0;
    if (!labels.empty()) {
        label = labels.back() + 1;
    }
    while (true) {
        cv::Mat frame;
        cv::Mat frame_gray;
        capture >> frame;
        if (!frame.empty())
        {
            
            cv::cvtColor( frame, frame_gray, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(frame_gray, frame_gray);
            
            face_cascade.detectMultiScale(frame_gray, faces );
            if (!faces.empty()) {
                
                cv::Mat facemat = frame_gray(faces[0]);

                //resizing the cropped image to suit to database image sizes
                cv::Mat face_resized;
                cv::resize(facemat, face_resized, cv::Size(128, 128), 0, 0, cv::INTER_CUBIC);
                trainingimages.push_back(face_resized);
                labels.push_back(label);
                numTrainIms++;
                
                cv::Point center( faces[0].x + faces[0].width/2, faces[0].y + faces[0].height/2 );
                cv::ellipse( frame, center, cv::Size( faces[0].width/2, faces[0].height/2 ), 0, 0, 360, cv::Scalar( 255, 0, 0 ), 4 );
    
                cv::imshow("imwindow", frame);
                
            }
            if (numTrainIms > 19)
            {
                cv::destroyWindow("imwindow");
                break;
            }
        }
        

        if(cv::waitKey(10) >= 0)
        {
            cv::destroyWindow("imwindow");
            break;
        }
    }

}

void MakePicture::testText()
{
//this will make a file but i dont know how to direct it to the right spot.
    //#####################
//    NSData *databuffer;
//    NSFileManager *filemgr = [NSFileManager defaultManager];
//    [filemgr createFileAtPath:@"newtester.txt" contents:nil attributes:nil];
    //#####################
    //this should get data from file. not tested yet.
//    databuffer = [filemgr contentsAtPath: @"test.txt" ];
    
    NSFileHandle *file;
            NSMutableData *data;

            const char *bytestring = "black dog";

            data = [NSMutableData dataWithBytes:bytestring length:strlen(bytestring)];


            file = [NSFileHandle fileHandleForUpdatingAtPath: @"/Users/ColtonShepherd/Documents/macopencvobjc/macopencvobjc/test.txt"];

            if (file == nil)
                    NSLog(@"Failed to open file");


//            [file seekToFileOffset: 10];

            [file writeData: data];

            [file closeFile];
    
    

    NSString *filterpath = [[NSBundle mainBundle] pathForResource:@"test"
                                                                    ofType:@"txt"];
    const CFIndex filter_len = 2048;
    char *filtername = (char *) malloc(filter_len);
    CFStringGetFileSystemRepresentation( (CFStringRef)filterpath, filtername, filter_len);
    std::ofstream filtersfile(filtername);

    int hello = 0;
    filtersfile << hello;
    std::cout << hello << std::endl;
    std::cout << "is this work \n";
}

//void MakePicture::trainface()
//{
//    NSString *faceCascadePath = [[NSBundle mainBundle] pathForResource:@"haarcascade_frontalface_alt2"
//                                                                    ofType:@"xml"];
//
//    const CFIndex CASCADE_NAME_LEN = 2048;
//    char *CASCADE_NAME = (char *) malloc(CASCADE_NAME_LEN);
//    CFStringGetFileSystemRepresentation( (CFStringRef)faceCascadePath, CASCADE_NAME, CASCADE_NAME_LEN);
//
//    if(!face_cascade.load(CASCADE_NAME))
//    {
//        std::cout << "face load failed";
//    }
//
//    cv::namedWindow("imwindow");
//    std::vector<cv::Rect> faces;
//    cv::VideoCapture capture(0);
//    int numTrainIms = 0;
//    int label = 0;
//    if (!labels.empty()) {
//        label = labels.back() + 1;
//    }
//    while (true) {
//        cv::Mat frame;
//        cv::Mat frame_gray;
//        capture >> frame;
//        if (!frame.empty())
//        {
//
//            cv::cvtColor( frame, frame_gray, cv::COLOR_BGR2GRAY);
//            cv::equalizeHist(frame_gray, frame_gray);
//
//            face_cascade.detectMultiScale(frame_gray, faces );
//            if (!faces.empty()) {
//
//                cv::Mat facemat = frame_gray(faces[0]);
//
//                //resizing the cropped image to suit to database image sizes
//                cv::Mat face_resized;
//                cv::resize(facemat, face_resized, cv::Size(128, 128), 0, 0, cv::INTER_CUBIC);
//                trainingimages.push_back(face_resized);
//                labels.push_back(label);
//                numTrainIms++;
//
//                cv::Point center( faces[0].x + faces[0].width/2, faces[0].y + faces[0].height/2 );
//                cv::ellipse( frame, center, cv::Size( faces[0].width/2, faces[0].height/2 ), 0, 0, 360, cv::Scalar( 255, 0, 0 ), 4 );
//
//                cv::imshow("imwindow", frame);
//
//            }
//            if (numTrainIms > 19)
//            {
//                cv::destroyWindow("imwindow");
//                break;
//            }
//        }
//
//
//        if(cv::waitKey(10) >= 0)
//        {
//            cv::destroyWindow("imwindow");
//            break;
//        }
//    }
//
//}



void MakePicture::recognizeFaces()
{
    Filters selectfilter;
    NSString *faceCascadePath = [[NSBundle mainBundle] pathForResource:@"haarcascade_frontalface_alt2"
                                                                    ofType:@"xml"];

    const CFIndex CASCADE_NAME_LEN = 2048;
    char *CASCADE_NAME = (char *) malloc(CASCADE_NAME_LEN);
    CFStringGetFileSystemRepresentation( (CFStringRef)faceCascadePath, CASCADE_NAME, CASCADE_NAME_LEN);

    if(!face_cascade.load(CASCADE_NAME))
    {
        std::cout << "face load failed";

    }
    cv::Mat framegray;
    std::vector<cv::Rect> faces;
    cv::namedWindow("imwindow");
    cv::moveWindow("imwindow", 40, 80);
    Facerecog face;
    
    svd = face.getsvd(trainingimages);
    cv::VideoCapture capture(0);
    while (true)
    {
        cv::Mat framegray;
        cv::Mat frame;
        std::vector<cv::Rect> faces;
        
        capture >> frame;
        if (!frame.empty())
        {
            cv::cvtColor(frame, framegray, cv::COLOR_BGR2GRAY);
            face_cascade.detectMultiScale( framegray, faces);
            
    //        -- Detect faces when not set to isblack.
            for (int i = 0; i < faces.size(); i++)
            {
                cv::Rect face_i = faces[i];

                //crop the roi from grya image
                cv::Mat facemat = framegray(face_i);

                //resizing the cropped image to suit to database image sizes
                cv::Mat face_resized;
                cv::resize(facemat, face_resized, cv::Size(128, 128), 0, 0, cv::INTER_CUBIC);
                cv::Mat dotpro = face.framemultiply(face_resized, svd);
                int smallestlabel = face.recognizefacedist(trainingimages, dotpro, svd, labels);
//                std::cout << "the label is " << smallestlabel << std::endl;

                if (filters[smallestlabel] == "eye swap") {
                    frame(face_i) = selectfilter.eyeswapfilter(frame(face_i));
                }

                else if (filters[smallestlabel] == "fire")
                {
                    frame = selectfilter.fireface(frame, face_i);
                }
                else if (filters[smallestlabel] == "cartoonify")
                {
                    frame(face_i) = selectfilter.cartoonify(frame(face_i));
                }
                else if (filters[smallestlabel] == "blackout")
                {
                    frame(face_i) = 0.0f;
                }

            }
            
            cv::imshow("imwindow", frame);
        }
        if(cv::waitKey(1) >= 0)
        {
            cv::destroyWindow("imwindow");
            break;
        }
    }
}


void MakePicture::makeblack(cv::Mat frame, std::vector<cv::Rect> faces)
{
    
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        cv::Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        cv::ellipse( frame, center, cv::Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, cv::Scalar( 0, 0, 0 ), -1 );

    }
}


void MakePicture::showFace()
{
    
    NSString *faceCascadePath = [[NSBundle mainBundle] pathForResource:@"haarcascade_frontalface_alt2"
                                                                    ofType:@"xml"];

    const CFIndex CASCADE_NAME_LEN = 2048;
    char *CASCADE_NAME = (char *) malloc(CASCADE_NAME_LEN);
    CFStringGetFileSystemRepresentation( (CFStringRef)faceCascadePath, CASCADE_NAME, CASCADE_NAME_LEN);

    if(!face_cascade.load(CASCADE_NAME))
    {
        std::cout << "face load failed";

    }
//    cv::Mat framegray;
    std::vector<cv::Rect> faces;
    cv::namedWindow("imwindow");
    cv::VideoCapture capture(0);
    while (true) {
        cv::Mat framegray;
        cv::Mat frame;
        std::vector<cv::Rect> faces;
        capture >> frame;
        if (!frame.empty()) {
            cv::cvtColor(frame, framegray, cv::COLOR_BGR2GRAY);
            face_cascade.detectMultiScale( framegray, faces);
            
    //        -- Detect faces when not set to isblack.
            if (!isblack)
            {

                for ( size_t i = 0; i < faces.size(); i++ )
                {
                    cv::Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
                    cv::ellipse( frame, center, cv::Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, cv::Scalar( 255, 0, 0 ), 4 );
                }
            }
            else if (isblack)
            {
                makeblack(frame, faces);
            }

            cv::imshow("imwindow", frame);
        }
        if(cv::waitKey(10) >= 0)
        {
            cv::destroyWindow("imwindow");
            break;
        }
        }
}


