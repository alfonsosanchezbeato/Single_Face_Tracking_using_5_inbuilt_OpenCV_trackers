/************************************************************************************************************************
 * Author     : Manu BN
 * Description: Detect face in the first frame using OpenCV's Haar cascade and track the same using 5 different inbuilt trackers namely :  BOOSTING, MIL, KCF, TLD and MEDIANFLOW
 ***********************************************************************************************************************/

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>


using namespace cv;
using namespace std;

// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

#define HAAR_DATA_DIR "/usr/share/opencv/haarcascades/"

bool track_frame(const string& trackerType, Rect2d& bbox, Tracker* tracker, Mat& frame)
{
    // Start timer
    double timer = (double)getTickCount();
    // Update the tracking result
    bool ok = tracker->update(frame, bbox);
    // Calculate Frames per second (FPS)
    float fps = getTickFrequency() / ((double)getTickCount() - timer);
    if (ok) {
        // Tracking success : Draw the tracked object
        rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
    } else {
        putText(frame, "Tracking failure detected", Point(100,80),
                FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
    }

    // Display tracker type and FPS on frame
    putText(frame, trackerType + " Tracker", Point(100,20),
            FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
    putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50),
            FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

    return ok;
}

bool detect_face(Tracker* tracker, Mat& frame, Rect2d& bbox)
{
    // Detect Face using Haar Cascade
    CascadeClassifier face_cascade;
    face_cascade.load(HAAR_DATA_DIR "haarcascade_frontalface_alt2.xml");
    std::vector<Rect> f;
    face_cascade.detectMultiScale(frame, f, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE);
    if (f.size() == 0)
        return false;

    // Define initial bounding box for the face detected in first frame i.e.
    // f[0]. Convert Rect co ordinates to Rect2D.
    Rect2d bbox_tmp(f[0].x, f[0].y, f[0].width, f[0].height);
    bbox = bbox_tmp;

    // Display bounding box.
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
    imshow("Tracking", frame);

    tracker->init(frame, bbox);

    return true;
}

int main(int argc, char **argv)
{
    // List of tracker types in OpenCV 3.2
    // NOTE : GOTURN implementation is buggy and does not work.
    string trackerTypes[6] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN"};
    // Create a tracker and select type by choosing indicies
    string trackerType = trackerTypes[4];

    Ptr<Tracker> tracker;

    #if (CV_MINOR_VERSION < 3)
    {
        tracker = Tracker::create(trackerType);
    }
    #else
    {
        if (trackerType == "BOOSTING")
            tracker = TrackerBoosting::create();
        if (trackerType == "MIL")
            tracker = TrackerMIL::create();
        if (trackerType == "KCF")
            tracker = TrackerKCF::create();
        if (trackerType == "TLD")
            tracker = TrackerTLD::create();
        if (trackerType == "MEDIANFLOW")
            tracker = TrackerMedianFlow::create();
        if (trackerType == "GOTURN")
            tracker = TrackerGOTURN::create();
    }
    #endif
    // Read video
    //VideoCapture video("mcem0_head.mpg");
    VideoCapture video(0);

    // Exit if video is not opened
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
    }

    Mat frame;
    Rect2d bbox;
    bool tracking = false;
    while(video.read(frame))
    {
        if (tracking)
            tracking = track_frame(trackerType, bbox, tracker, frame);

        if (!tracking)
            tracking = detect_face(tracker, frame, bbox);

        // Display frame.
        imshow("Tracking", frame);

        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
            break;
    }
}
