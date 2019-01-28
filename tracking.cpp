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

#define HAAR_DATA_DIR "/usr/share/opencv/haarcascades/"

using namespace cv;
using namespace std;

bool track_frame(const Mat& frame, Tracker& tracker, Rect2d& bbox)
{
    return tracker.update(frame, bbox);
}

bool detect_face(const Mat& frame, Tracker& tracker, Rect2d& bbox)
{
    // Detect face using Haar Cascade classifier
    CascadeClassifier face_cascade;
    face_cascade.load(HAAR_DATA_DIR "haarcascade_frontalface_alt2.xml");
    vector<Rect> f;
    face_cascade.detectMultiScale(frame, f, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE);
    if (f.size() == 0)
        return false;

    cout << "Detected " << f.size() << " faces \n";
    // Get only one face for the moment
    bbox = Rect2d(f[0].x, f[0].y, f[0].width, f[0].height);

    tracker.init(frame, bbox);

    return true;
}

class TrackData {
public:
    Rect2d bbox;
    bool tracking;
};

void detect_and_track_loop(atomic_bool &finish, mutex &frameMtx,
                           condition_variable &frameCondition, Mat &frame,
                           TrackData &trackData)
{
    // List of tracker types in OpenCV 3.2
    // NOTE : GOTURN implementation is buggy and does not work.
    string trackerTypes[6] = {"BOOSTING", "MIL", "KCF", "TLD",
                              "MEDIANFLOW", "GOTURN"};
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
            std::unique_lock<mutex> lock(frameMtx);
            frameCondition.wait(lock);

            if (tracking)
                tracking = track_frame(frame, *tracker, bbox);

            if (!tracking)
                tracking = detect_face(frame, *tracker, bbox);

            trackData.bbox = bbox;
            trackData.tracking = tracking;

            cout << "Size is " << frame.cols << " x " << frame.rows
                 << " . Tracking: " << tracking << '\n';
        }

        if (finish)
            break;
    }
}

int main(int argc, char **argv)
{
    // Read video from either camera of video file
    VideoCapture video;
    if (argc == 1) {
        video.open(0);
    } else if (argc == 2) {
        int videoSrc;
        istringstream arg1(argv[1]);
        arg1 >> videoSrc;
        video.open(videoSrc);
    } else if (argc == 3 && strcmp(argv[1], "-f") == 0) {
        video.open(argv[2]);
    } else {
        cout << "Usage: " << argv[0] << " [<dev_number> | -f <video_file>]\n";
        return 1;
    }

    // Exit if video is not opened
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
    }

    mutex frameMtx;
    atomic_bool finish{false};
    Mat frame;
    TrackData trackData;
    condition_variable frameCondition;
    thread detectAndTrack(detect_and_track_loop, ref(finish), ref(frameMtx),
                          ref(frameCondition), ref(frame), ref(trackData));

    double scale_f = 2.;
    Mat frameLast;
    while(video.read(frameLast))
    {
        {
            std::unique_lock<mutex> lock(frameMtx, defer_lock_t());
            if (lock.try_lock()) {
                // Take latest track result
                if (trackData.tracking) {
                    Rect2d bbox(scale_f*trackData.bbox.x,
                                scale_f*trackData.bbox.y,
                                scale_f*trackData.bbox.width,
                                scale_f*trackData.bbox.height);
                    rectangle(frameLast, bbox, Scalar(255, 0, 0), 2, 1);
                }
                // Makes a copy to the shared frame
                resize(frameLast, frame, cv::Size(), 1/scale_f, 1/scale_f);
                frameCondition.notify_one();
            }
        }

        imshow("Tracking", frameLast);

        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
            break;
    }

    finish = true;
    frameCondition.notify_one();
    detectAndTrack.join();
}
