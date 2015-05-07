//
//  DatasetVOT2014.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 1/22/15.
//
//

#include "DatasetVOT2014.h"
#include <fstream>
#include <regex>
#include <math.h>



std::vector<std::pair<std::string,std::vector<std::string>>> DatasetVOT2014::prepareDataset(std::string rootFolder){

    using namespace std;

    vector<pair<string, vector<string>>> video_gt_images;


    // step #1: list all videos


    vector<string> videos=listSubFolders(rootFolder);


    for (int i=0; i<(int)videos.size(); i++) {

        // create a string which represents ground truth and image location
        string imgLocation=rootFolder+videos[i]+"/";


        string gt=rootFolder+videos[i]+"/groundtruth.txt";

        // list all images

        this->vidToIndex.insert(std::pair<std::string, int>(videos[i],i));
        vector<string> images=Dataset::listImages(imgLocation, "jpg");

        std::pair<string, vector<string>> groundTruth_images;
        groundTruth_images=std::make_pair(gt, images);

        video_gt_images.push_back(groundTruth_images);

        this->videos.push_back(videos[i]);

    }


    return video_gt_images;
}

float dist(cv::Point2f x1,cv::Point2f x2){
    return sqrt(pow(x1.x-x2.x,2)+pow(x1.y-x2.y,2));
}

std::vector<cv::Rect> DatasetVOT2014::readGroundTruth(std::string fileName){
     std::vector<cv::Rect> gtRect;

    std::vector<cv::RotatedRect> rotatedRects=readComplete(fileName);

    for (int i=0; i<rotatedRects.size(); i++) {

        //cv::Rect r=rotatedRects[i].boundingRect();

        cv::Point2f center=rotatedRects[i].center;

        cv::Point2f size=rotatedRects[i].size;

        cv::Rect r(center.x-size.x/2.0,center.y-size.y/2,size.x,size.y);
        //cv::Rect r(center.x-size.y/2.0,center.y-size.x/2,size.y,size.x);
        //r=rotatedRects[i].boundingRect();
        gtRect.push_back(r);

    }

    return gtRect;
}


cv::RotatedRect DatasetVOT2014::constructRotatedRect(std::vector<float> record) {

    //1) find center -> simply average

    float center_x=0;
    float center_y=0;


    cv::Point2f vertices[4];
    for (int i=0; i<4; i++) {
        center_x+=record[i*2];
        center_y+=record[i*2+1];

        vertices[i].x=record[i*2];
        vertices[i].y=record[i*2+1];
    }

    center_x/=4;
    center_y/=4;

    //2) find width and height ( find for both sides and average)


    // (abs(X2-X1)+abs(X4-X3))/2
    float width=sqrtf(powf((vertices[2].x-vertices[1].x),2)+powf((vertices[2].y-vertices[1].y),2));
    //float height=dist(vertices[2],vertices[0]);

    double w1=dist(vertices[0],vertices[1]);

    //float height=sqrtf(powf((vertices[0].x-vertices[2].x),2)+powf((vertices[0].y-vertices[2].y),2));
    float height=dist(vertices[0],vertices[1]);
    //3) find the angle

    float angle=findAngle(vertices);

    cv::RotatedRect rot_rect(cv::Point2f(center_x,center_y),cv::Size2f(width,height),angle);

    return rot_rect;
}

std::vector<cv::RotatedRect> DatasetVOT2014::readComplete(std::string fileName){


    using namespace std;


    vector<cv::RotatedRect> gtRect;

    std::ifstream infile(fileName);
    string str;
    std::regex e("[[:digit:]]+\\.[[:digit:]]+");

    int idx=0;

    float num=0;
    while (std::getline(infile, str))
    {


        std::regex_iterator<std::string::iterator> rit ( str.begin(), str.end(), e );
        std::regex_iterator<std::string::iterator> rend;



        std::vector<float> record;
        while (rit!=rend) {


            num=stof(rit->str());

            record.push_back(num);
            ++rit;
        }

        // every rectangle is represented as X1, Y1, X2, Y2, X3, Y3, X4, Y4 stored in the vector 'record'



        cv::RotatedRect rot_rect=constructRotatedRect(record);
        gtRect.push_back(rot_rect);

        ++idx;
    }


    return gtRect;

}


void DatasetVOT2014::showVideo(std::string rootFolder, int vidNumber){

    std::vector<std::pair<std::string,std::vector<std::string>>> set=prepareDataset(rootFolder);
    using namespace std;
    using namespace cv;

    if (vidNumber<0 || vidNumber>=set.size()) {
        cout<<"Wrong video ID. The ID requested is out of bounds for this dataset."<<endl;
        return;
    }

    vector<cv::RotatedRect> r= readComplete(set[vidNumber].first);


    vector<string> images=set[vidNumber].second;


    for (int frame=0; frame<r.size(); frame++) {

        cv::Mat M,rotated,cropped;
        RotatedRect rect=r[frame];

        float angle=r[frame].angle;

        //std::cout<<"Angle: "<<angle<<std::endl;
        cv::Size rect_size=r[frame].size;

//        if (rect.angle < -45.) {
//            angle += 90.0;
//            swap(rect_size.width, rect_size.height);
//        }

        cv::Mat im=cv::imread(images[frame]);



        // get the rotation matrix
        M = getRotationMatrix2D(rect.center, angle, 1.0);
        // perform the affine transformation
        warpAffine(im, rotated, M, im.size(), INTER_CUBIC);
        // crop the resulting image
        getRectSubPix(rotated, rect_size, rect.center, cropped);

        Point2f vertices[4];

        rect.points(vertices);
        for (int i = 0; i < 4; i++){
            line(im, vertices[i], vertices[(i+1)%4], Scalar(0,255,0));

        }

        cv::Rect regular=rect.boundingRect();

//        rectangle(im, regular, cv::Scalar(150));
//        cv::imshow("cropped", cropped);
//        cv::waitKey(2);

        cv::imshow("original",im);
        cv::waitKey(30);
    }

}




float DatasetVOT2014::findAngle(cv::Point2f pts[]){


//    int i=0;
//    double b=dist(pts[i],pts[i+1]);
//    double c=dist(pts[i+1],pts[i+2]);
//    double a=dist(pts[i],pts[i+2]);
//
//
//
//    float angle=(180/3.14159)*(std::acos((b*b+c*c-a*a)/(2*b*c)))-90;


    //1 find center

    double center_x=0;
    double center_y=0;

    for (int j = 0; j < 4; ++j) {
        center_x+=pts[j].x;
        center_y+=pts[j].y;
    }

    center_x=center_x/4.0;
    center_y=center_y/4.0;


    double width=dist(pts[2],pts[1]);
    double height=dist(pts[0],pts[1]);


    arma::mat x_new(2,2,arma::fill::zeros);

    arma::mat x_old(2,2,arma::fill::zeros);

    x_new(0,0)=-width/2;
    x_new(1,0)=-height/2;
    x_new(0,1)=width/2;
    x_new(1,1)=-height/2;


    cv::Point2f leftmost=pts[0];
    leftmost.x=leftmost.x-center_x;
    leftmost.y=leftmost.y-center_y;
    cv::Point2f uppermost=pts[3];

    uppermost.x=uppermost.x-center_x;
    uppermost.y=uppermost.y-center_y;
//    for (int k = 0; k < 4; ++k) {
//        if (leftmost.x<pts[k].x){
//            leftmost=pts[k];
//        }
//
//        if(uppermost.y<pts[k].y){
//            uppermost=pts[k];
//        }
//    }

    x_old(0,0)=leftmost.x;
    x_old(1,0)=leftmost.y;
    x_old(0,1)=uppermost.x;
    x_old(1,1)=uppermost.y;


    arma::mat rho=x_new*inv(x_old);


    // std::cout<<rho<<std::endl;



    double angle=(180/M_PI)*std::acos(rho(0,0));
//
//
//    float angleinDegree=atan2f(deltaX, deltaY)*(180/M_PI);

    return angle;

}



