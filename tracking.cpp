/*******************************************************************************
 * Author     : Manu BN, Alfonso Sanchez-Beato
 * Description: Detect face in the first frame using OpenCV's Haar cascade and
 * track the same using 5 different inbuilt trackers namely :
 * BOOSTING, MIL, KCF, TLD and MEDIANFLOW
 ******************************************************************************/
#include <atomic>
#include <future>
#include <mutex>
#include <thread>
#include <utility>

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

// mutex frameMtx_g, trackDataMtx_g;
// atomic_bool cv_work_busy_g, finish_g;
// Mat frame_g;
// Rect2d bbox_g;
// condition_variable frameCondition_g;

// class OpenCVThread
// {
// public:
//     OpenCVThread(atomic_bool &cv_work_busy, atomic_bool &finish,
//                     mutex &frameMtx, condition_variable &frameCondition,
//                     Mat &frame, mutex &trackDataMtx, Rect2d &bbox)
//     {
//         cv_work_busy_ = cv_work_busy;
//         finish_ = finish;
//         frameMtx_ = frameMtx;
//         frameCondition_ = frameCondition;
//         frame_ = frame;
//         trackDataMtx_ = trackDataMtx;
//         bbox_ = bbox;
//     }

// private:
//     mutex &frameMtx_, &trackDataMtx_;
//     atomic_bool &cv_work_busy_, &finish_;
//     Mat &frame_;
//     Rect2d &bbox_;
//     condition_variable &frameCondition_;
// };


class TrackData {
public:
    mutex mtx;
    Rect2d bbox;
    bool tracking;
};

void detect_and_track_loop(atomic_bool &cv_work_busy_g, atomic_bool &finish_g,
                           mutex &frameMtx_g, condition_variable &frameCondition_g,
                           Mat &frame_g, TrackData &trackData)
{
    // List of tracker types in OpenCV 3.2
    // NOTE : GOTURN implementation is buggy and does not work.
    string trackerTypes[6] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN"};
    // Create a tracker and select type by choosing indicies
    string trackerType = trackerTypes[4]; // MEDIANFLOW

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

    bool tracking = false;
    Rect2d bbox;

    while (true) {
        {
            std::unique_lock<mutex> lock(frameMtx_g);
            frameCondition_g.wait(lock);
            cv_work_busy_g = true;

            if (tracking)
                tracking = track_frame(trackerType, bbox, tracker, frame_g);

            if (!tracking)
                tracking = detect_face(tracker, frame_g, bbox);

            cout << "Size is " << frame_g.cols << " x " << frame_g.rows
                 << " . Tracking: " << tracking << endl;

            cv_work_busy_g = false;
        }

        if (trackData.bbox != bbox || trackData.tracking != tracking) {
            lock_guard<decltype(trackData.mtx)> lock(trackData.mtx);
            trackData.bbox = bbox;
            trackData.tracking = tracking;
        }

        if (finish_g)
            break;
    }
}

int main(int argc, char **argv)
{
    // Read video
    //VideoCapture video("mcem0_head.mpg");
    VideoCapture video(0);

    // Exit if video is not opened
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
    }

    mutex frameMtx_g;
    atomic_bool cv_work_busy_g{false}, finish_g{false};
    Mat frame_g;
    TrackData trackData;
    condition_variable frameCondition_g;
    thread detectAndTrack(detect_and_track_loop, ref(cv_work_busy_g), ref(finish_g),
                          ref(frameMtx_g), ref(frameCondition_g),
                          ref(frame_g), ref(trackData));
    double scale_f = 2.;

    Mat frame;
    while(video.read(frame))
    {
        if (!cv_work_busy_g) {
            lock_guard<decltype(frameMtx_g)> lock(frameMtx_g);
            // Makes a copy to the shared frame
            resize(frame, frame_g, cv::Size(), 1/scale_f, 1/scale_f);
            frameCondition_g.notify_one();
        }

        {
            lock_guard<decltype(trackData.mtx)> lock(trackData.mtx);
            if (trackData.tracking) {
                Rect2d bbox(scale_f*trackData.bbox.x, scale_f*trackData.bbox.y,
                            scale_f*trackData.bbox.width, scale_f*trackData.bbox.height);
                rectangle(frame, bbox, Scalar( 255, 0, 0), 2, 1);
            }
        }

        imshow("Tracking", frame);

        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
            break;
    }

    finish_g = true;
    detectAndTrack.join();
}
