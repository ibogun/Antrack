//
//  Experiment.cpp
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/15/15.
//
//

#include "Experiment.h"


std::vector<std::tuple<int, int, cv::Rect>> Experiment::generateAllBoxesToEvaluate(Dataset *dataset) {
    using namespace std;
    vector<pair<string, vector<string>>> video_gt_images =
            dataset->prepareDataset();


    std::vector<std::tuple<int, int, cv::Rect>> exp_boxes;// video - frame - bbox

    for (int i = 0; i < video_gt_images.size(); ++i) {

        pair<string, vector<string>> gt_images = video_gt_images[i];
        vector<cv::Rect> groundTruth = dataset->readGroundTruth(gt_images.first);

        cv::Mat image = cv::imread(gt_images.second[0]);
        int n = image.cols;
        int m = image.rows;

        std::vector<std::pair<cv::Rect, int>> experiment_boxes = this->generateBoundingBoxes(groundTruth, n, m);

        for (int j = 0; j < experiment_boxes.size(); ++j) {

            auto t = make_tuple(i, experiment_boxes[j].second, experiment_boxes[j].first);
            exp_boxes.push_back(t);
        }

    }

    return exp_boxes;

}


void Experiment::showBoxes(Dataset *dataset, int vidNumber) {
    using namespace std;
    std::vector<std::tuple<int, int, cv::Rect>> b=this->generateAllBoxesToEvaluate(dataset);

    vector<pair<string, vector<string>>> video_gt_images =
                                dataset->prepareDataset();

    for (int j = 0; j < b.size(); ++j) {
        if (std::get<0>(b[j])==vidNumber){
            int frame=std::get<1>(b[j]);
            cv::Rect r=std::get<2>(b[j]);
            cv::Mat im=cv::imread(video_gt_images[vidNumber].second[frame]);

            cv::rectangle(im,r,cv::Scalar(144,144,144),2);

            cv::imshow("",im);

            cv::waitKey();
            cv::destroyAllWindows();
        }
    }
}


//std::ostream& operator<<(std::ostream &strm, const Experiment &f){
//    return strm;
//}
