//
//  ViewController.m
//  macopencvobjc
//
//  Created by Colton Shepherd on 3/23/22.
//
#include "imagetest.hpp"
#import "ViewController.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>

@implementation ViewController
MakePicture makeimage;

//training button
- (IBAction)imageButton:(NSButton *)sender {
    makeimage.showPicture();
}
//recognition button after training
- (IBAction)faceRecog:(NSButton *)sender {
    makeimage.recognizeFaces();
}
- (IBAction)filtermenu:(NSPopUpButton *)sender {
    if([sender.selectedItem.title isEqual: @"blackout"])
    {
        const char *filterchar = [sender.selectedItem.title UTF8String];
        makeimage.setfiltername(filterchar);
    }
    if([sender.selectedItem.title isEqual: @"eye swap"])
    {
        const char *filterchar = [sender.selectedItem.title UTF8String];
        makeimage.setfiltername(filterchar);
        
    }
    if([sender.selectedItem.title isEqual: @"cartoonify"])
    {
        const char *filterchar = [sender.selectedItem.title UTF8String];
        makeimage.setfiltername(filterchar);
    }
}




- (IBAction)faceButton:(NSButton *)sender {
    makeimage.showFace();
//    makeimage.testText();
}
- (IBAction)hideButton:(NSButton *)sender {
    makeimage.setblack();
//    hideButton:sender.title = @"show face";
    if ([sender.title  isEqual: @"hide face"])
    {
        sender.title = @"show face";
    }
    else if ([sender.title  isEqual: @"show face"])
    {
        sender.title = @"hide face";
    }
}


- (void)viewDidLoad {
    [super viewDidLoad];
    
    
    // Do any additional setup after loading the view.
}


- (void)setRepresentedObject:(id)representedObject {
    [super setRepresentedObject:representedObject];

    // Update the view, if already loaded.
}


@end
