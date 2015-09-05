//
//  DatasetVOT2015.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 1/22/15.
//
//

#include "DatasetVOT2015.h"
#include <fstream>
#include <regex>
#include <math.h>


cv::Rect DatasetVOT2015::constructRectangle(std::vector<float> record) {
    //1) find center -> simply average

    float center_x=0;
    float center_y=0;

    std::vector<float> x_coordinates;
    std::vector<float> y_coordinates;

    for (int i=0; i<4; i++) {
        center_x+=record[i*2];
        center_y+=record[i*2+1];
        x_coordinates.push_back(record[i*2]);
        y_coordinates.push_back(record[i*2+1]);
    }

    std::sort(x_coordinates.begin(), x_coordinates.end());
    std::sort(y_coordinates.begin(), y_coordinates.end());

    float width = ((x_coordinates[2] + x_coordinates[3]) -
                   (x_coordinates[0] + x_coordinates[1]))/2;

    float height = ((y_coordinates[2] + y_coordinates[3]) -
                    (y_coordinates[0] + y_coordinates[1]))/2;


    center_x/=4;
    center_y/=4;

    float top_left_x = center_x - width / 2.0;
    float top_left_y = center_y - height / 2.0;

    cv::Rect r(int(top_left_x), int(top_left_y), width, height);

    return r;
}


std::vector<std::vector<float>> DatasetVOT2015::readAllRecords(std::string fileName) {


    using namespace std;


    vector<vector<float>> gtRect;

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

        gtRect.push_back(record);

        ++idx;
    }


    return gtRect;
}

std::vector<cv::Rect> DatasetVOT2015::readGroundTruth(std::string fileName){
     std::vector<cv::Rect> gtRect;

    std::vector<std::vector<float>> floats = readAllRecords(fileName);

    for (int i=0; i<floats.size(); i++) {

        cv::Rect r = this->constructRectangle(floats[i]);
        gtRect.push_back(r);

    }

    return gtRect;
}