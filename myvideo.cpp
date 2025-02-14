#include <opencv2/opencv.hpp>
#include "System.h"

#include <string>
#include <chrono> // for time stamp
#include <iostream>
#include <fstream> // for file output

using namespace std;

// Paths to files
string parameterFile = "/home/autonomy/Dev/ORB_SLAM3/Examples/Monocular/myvideo.yaml";
string vocFile = "/home/autonomy/Dev/ORB_SLAM3/Vocabulary/ORBvoc.txt";
string videoFile = "/home/autonomy/Dev/ORB_SLAM3/Examples/Monocular/lab_video1.mp4";

void ORB_SLAM3::System::SavePointCloud(const string &filename){
    // Code is based on MapDrawer::DrawMapPoints()
    cout << endl << "Saving map point coordinates to " << filename << " ..." << endl;
    cout << endl << "Number of maps is: " << mpAtlas->CountMaps() << endl;

    // TODO Get all maps or is the current active map is enough?
    // vector<Map*> vpAllMaps = mpAtlas->GetAllMaps()

    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if(!pActiveMap) {
        cout << endl << "There is no active map (pActiveMap is null)" << endl;
        return;
    }

    // Vectors containing pointers to MapPoint objects contained in the maps
    // Vector of pointers for Map Points -- vpMPs
    // Vector of pointers for Reference Map Points -- vpRefMPs
    // TODO figure out the difference between Reference Map Points and normal Map Points
    const vector<MapPoint*> &vpMPs = pActiveMap->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = pActiveMap->GetReferenceMapPoints();

    if(vpMPs.empty()){
        cout << endl << "Vector of map points vpMPs is empty!" << endl;
        return;
    }

    // Use a set for fast lookup of reference frames
    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    // Get the output file stream in fixed-point format for map points
    ofstream f;
    f << "pos_x, pos_y, pos_z";
    f.open(filename.c_str());
    f << fixed;

    // TODO figure out if we need to consider whether the presence of IMU
    // requires some transforms/exceptions

    // Iterate over map points, skip "bad" ones and reference map points
    for (size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if (vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i])){
            continue;
        }
        Eigen::Matrix<float,3,1> pos = vpMPs[i]->GetWorldPos();
        f << pos(0) << ", " << pos(1) << ", " << pos(2) << endl;
    }

    // Close the output stream
    f.close();

    // Get the output file stream in fixed-point format for reference map points
    f.open(("ref_" + filename).c_str());
    f << "pos_x, pos_y, pos_z" << endl;
    f << fixed;

    // Iterate over reference map points, skip if bad
    for (set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad()){
            continue;
        }
        Eigen::Matrix<float,3,1> pos = (*sit)->GetWorldPos();
        f << pos(0) << ", " << pos(1) << ", " << pos(2) << endl;
    }

    // Close the output stream
    f.close();
}

int main(int argc, char **argv) {
    // Initialize ORB-SLAM3 system
    ORB_SLAM3::System SLAM(vocFile, parameterFile, ORB_SLAM3::System::MONOCULAR, true);

    // Open video file
    cv::VideoCapture cap(videoFile); // change to 1 if you want to use USB camera.
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video file." << endl;
        return -1;
    }


    auto start = chrono::system_clock::now();


    while (1) {
        cv::Mat frame;
        cap >> frame;
        if ( frame.data == nullptr )
        break;

        // rescale because image is too large
        cv::Mat frame_resized;
        cv::resize(frame, frame_resized, cv::Size(640,480));

        auto now = chrono::system_clock::now();
        auto timestamp = chrono::duration_cast<chrono::milliseconds>(now - start);
        SLAM.TrackMonocular(frame_resized, double(timestamp.count())/1000.0);
        cv::waitKey(30);
    }
    // Save the keyframe trajectory in TUM format
    //SLAM.SaveKeyFrameTrajectoryTUM("/home/autonomy/Dev/YOLO_Py/trajectory/MyVideoKeyFrameTrajectoryTUMFormat.txt");
    SLAM.SaveTrajectoryTUM("/home/autonomy/Dev/YOLO_Py/trajectory/CameraTrajectory.txt"); // cannot be used for monocular camera
    SLAM.SavePointCloud("/home/autonomy/Dev/YOLO_Py/point_cloud/PointCloud.txt");
    // SLAM.SaveTrajectoryTUM("MyVideoFrameTrajectoryTUMFormat.txt"); // cannot be used for monocular camera

    //system("python3 /home/autonomy/Dev/YOLO_Py/plot_and_save_keyframes.py");


    // Shutdown ORB-SLAM3
    SLAM.Shutdown();

    return 0;
}