//
//  facefilters.cpp
//  macopencvobjc
//
//  Created by Colton Shepherd on 4/30/22.
//

#include "facefilters.hpp"
#import "ViewController.h"

Filters::Filters()
{
    j = 0;
    x=0;
}

cv::Mat Filters::cartoonify(cv::Mat face)
{
    
    cv::Mat gray;
    cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
    const int median_blur_filter_size = 7;
    cv::medianBlur(gray, gray, median_blur_filter_size);
    cv::Mat edges;
    const int laplacian_filter_size = 5;
    cv::Laplacian(gray, edges, CV_8U, laplacian_filter_size);
    cv::Mat mask;
    const int edges_threshold = 80;
    cv::threshold(edges, mask, edges_threshold, 255, cv::THRESH_BINARY_INV);
    cv::Mat tmp;
    
    int repetitions = 14;
    for (int i = 0; i < repetitions; i++) {
        int ksize = 9;
        double colorstrength = 7;
        double spatialstrength = 12;
        cv::bilateralFilter(face, tmp, ksize, colorstrength, spatialstrength);
        cv::bilateralFilter(tmp, face, ksize, colorstrength, spatialstrength);
    }
    cv::Mat dst;
    dst.setTo(0);
    face.copyTo(dst, mask);

    return dst;
}

void Filters::makeblack(cv::Mat frame, std::vector<cv::Rect> faces)
{
    
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        cv::Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        cv::ellipse( frame, center, cv::Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, cv::Scalar( 0, 0, 0 ), -1 );

    }
}

cv::Mat Filters::fireface(cv::Mat frame, cv::Rect face_rect)
{
    
    
    NSString *firepath = [[NSBundle mainBundle] pathForResource:@"fireanimation"
                                                                    ofType:@"jpg"];
    const CFIndex fire_length = 2048;
    char *fireanimation = (char *) malloc(fire_length);
    CFStringGetFileSystemRepresentation( (CFStringRef)firepath, fireanimation, fire_length);
    cv::Mat fireanimationreel = cv::imread(fireanimation);
    
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
    cv::resize(animation, animationcopy, cv::Size(face_rect.size().width, face_rect.size().height/2));
    cv::Rect face_i_copy(face_rect.tl().x, face_rect.tl().y - 100, face_rect.width, face_rect.height/2);

    cv::Mat animationgray, finalanim;
    cv::cvtColor(animationcopy, animationgray, cv::COLOR_BGR2GRAY);
    cv::threshold(animationgray, animationgray, 150.0, 255, cv::THRESH_TOZERO);
    animationcopy.copyTo(finalanim, animationgray);


    //blend images.
    bitwise_or(frame(face_i_copy), finalanim, frame(face_i_copy));
//  animationcopy.copyTo(firelocation);
    
    animation.release();
    j++;
    return frame;
}
        

cv::Mat Filters::eyeswapfilter(cv::Mat face)
{
    cv::CascadeClassifier eyes_cascade;
    cv::CascadeClassifier mouth_cascade;
    NSString *eyecascadepath = [[NSBundle mainBundle] pathForResource:@"haarcascade_eye"
                                                                    ofType:@"xml"];
    const CFIndex eyeCASCADE_NAME_LEN = 2048;
    char *eyeCASCADE_NAME = (char *) malloc(eyeCASCADE_NAME_LEN);
    CFStringGetFileSystemRepresentation( (CFStringRef)eyecascadepath, eyeCASCADE_NAME, eyeCASCADE_NAME_LEN);

    NSString *mouthCascadePath = [[NSBundle mainBundle] pathForResource:@"haarcascade_smile"
                                                                    ofType:@"xml"];
    const CFIndex mouthCASCADE_NAME_LEN = 2048;
    char *mouthCASCADE_NAME = (char *) malloc(mouthCASCADE_NAME_LEN);
    CFStringGetFileSystemRepresentation( (CFStringRef)mouthCascadePath, mouthCASCADE_NAME, mouthCASCADE_NAME_LEN);

    
    if (!eyes_cascade.load(eyeCASCADE_NAME)) {
        std::cout << " Error loading eyes" << std::endl;
        return face;
    }
    // mouth/smile detection
    if (!mouth_cascade.load(mouthCASCADE_NAME)) {
        std::cout << " Error loading mouth" << std::endl;
        return face;
    }
    
    
    
    std::vector<cv::Rect> eyes;
    std::vector<cv::Rect> mouth;
    //detect mouth and eyes.
    eyes_cascade.detectMultiScale(face, eyes, 1.1, 3, 0, cv::Size(40, 40));
    mouth_cascade.detectMultiScale(face, mouth, 1.1, 3, 0, cv::Size(100, 100));
    std::cout << "the eyes list size is: " << std::to_string(eyes.size()) << std::endl;
    std::cout << "the mouth list size is: " << std::to_string(mouth.size()) << std::endl;
    if (eyes.size() && mouth.size()) {
        //create rectangles so mouth and eyes can copy onto each other.
        cv::Rect eyerec(eyes[0].tl().x, eyes[0].tl().y, eyes[0].width, eyes[0].height);
        cv::Rect mouthrec(mouth[0].tl().x, mouth[0].tl().y, mouth[0].width, mouth[0].height);
        
        cv::Mat eyeImage = face(eyerec);
        cv::Mat mouthImage = face(mouthrec).clone();
        cv::Mat copyeyeImage, copymouthImage;
        cv::resize(eyeImage, copyeyeImage, cv::Size(mouthImage.cols, mouthImage.rows));
        cv::resize(mouthImage, copymouthImage, cv::Size(eyeImage.cols, eyeImage.rows));
        
        
        
//          this section for trying to make ellipse animations.

        
//                        rectangles for mask
        cv::Mat eye_ellipticmask = eyeImage.clone();
        cvtColor(eye_ellipticmask, eye_ellipticmask, cv::COLOR_BGR2GRAY);
        cv::RotatedRect ellipsedeye;
        ellipsedeye.center = cv::Point2f(eyeImage.cols/2, eyeImage.rows/2);
        ellipsedeye.size = cv::Size(eyeImage.cols, eyeImage.rows);
        
        cv::Mat mouth_ellipticmask = mouthImage.clone();
        cv::cvtColor(mouth_ellipticmask, mouth_ellipticmask, cv::COLOR_BGR2GRAY);
        cv::RotatedRect ellipsedmouth;
        ellipsedmouth.center = cv::Point2f(mouthImage.cols/2, mouthImage.rows/2);
        ellipsedmouth.size = cv::Size(mouthImage.cols, mouthImage.rows);

//                        make mask
        cv::ellipse(eye_ellipticmask, ellipsedeye, CV_RGB(255, 255, 255), -1);
        cv::ellipse(mouth_ellipticmask, ellipsedmouth, CV_RGB(255, 255, 255), -1);
        cv::threshold(eye_ellipticmask, eye_ellipticmask, 250, 255, cv::THRESH_TOZERO);
        cv::threshold(mouth_ellipticmask, mouth_ellipticmask, 250, 255, cv::THRESH_TOZERO);
//                        copy mask onto image.

        cv::Mat mouthlocation(face, mouthrec);
        cv::Mat eyelocation(face, eyerec);
        copyeyeImage.copyTo(mouthlocation, mouth_ellipticmask);
        copymouthImage.copyTo(eyelocation, eye_ellipticmask);
        eyeImage.release();
        copymouthImage.release();
        
        //if second eye detected copy mouth onto second eye.
        if (eyes.size() > 1) {
            cv::Rect eyerec2(eyes[1].tl().x, eyes[1].tl().y, eyes[1].width, eyes[1].height);
            eyeImage = face(eyerec2);
            cv::resize(mouthImage, copymouthImage, cv::Size(eyeImage.cols, eyeImage.rows));
            
//                            second eye mask
            eye_ellipticmask = eyeImage.clone();
            cv::cvtColor(eye_ellipticmask, eye_ellipticmask, cv::COLOR_BGR2GRAY);
            cv::RotatedRect ellipsedeye2;
            ellipsedeye2.center = cv::Point2f(eyeImage.cols/2, eyeImage.rows/2);
            ellipsedeye2.size = cv::Size(eyeImage.cols, eyeImage.rows);
            cv::ellipse(eye_ellipticmask, ellipsedeye, CV_RGB(255, 255, 255), -1);
            threshold(eye_ellipticmask, eye_ellipticmask, 250, 255, cv::THRESH_TOZERO);
            
            
            cv::Mat eye2location(face, eyerec2);
            copymouthImage.copyTo(eye2location, eye_ellipticmask);
        }
    } else
    {
        std::cout << "not detecting mouth or eyes. \n";
    }
    
    return face;
}


Filters::~Filters()
{
    
}
