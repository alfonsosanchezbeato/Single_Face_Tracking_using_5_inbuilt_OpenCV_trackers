/*******************************************************************************
 * Author     : Alfonso Sanchez-Beato, based on example from Manu BN
 * Description: Detect face using OpenCV's Haar cascade and track it until lost.
 *  At that point, try to detect face again and track in loop.
 ******************************************************************************/
#include <mutex>
#include <thread>
#include <condition_variable>

#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

using namespace cv;
using namespace std;

// Detects faces on video frames
struct FaceDetector {
    FaceDetector(void);
    // If a face is detected in frame, returs true and fills bbox with a square
    // containing it.
    bool detectFace(const Mat& frame, Rect2d& bbox);

private:
    bool debug;
    CascadeClassifier faceCascade;
};

FaceDetector::FaceDetector(void) :
    debug{getenv("FACETRACKER_DEBUG") ? true : false}
{
    string pathHaarData;
    const char *snapDir = getenv("SNAP");

    // We make sure things work well if we are in a snap
    if (snapDir) {
        pathHaarData.append(snapDir);
        pathHaarData.append("/usr/share/opencv4/haarcascades/");
    } else {
        pathHaarData.append("/usr/share/opencv/haarcascades/");
    }
    pathHaarData.append("haarcascade_frontalface_alt2.xml");

    faceCascade.load(pathHaarData);
}

bool FaceDetector::detectFace(const Mat& frame, Rect2d& bbox)
{
    // Detect face using Haar Cascade classifier
    // See http://www.emgu.com/wiki/files/1.5.0.0/Help/html/e2278977-87ea-8fa9-b78e-0e52cfe6406a.htm
    // for flag description. It might be wortwhile to play a bit with the
    // different parameters.
    vector<Rect> f;
    faceCascade.detectMultiScale(frame, f, 1.1, 2, CASCADE_SCALE_IMAGE);
    if (f.size() == 0)
        return false;

    if (debug)
        cout << "Detected " << f.size() << " faces \n";

    // Get only one face for the moment
    bbox = Rect2d(f[0].x, f[0].y, f[0].width, f[0].height);

    return true;
}

// Detects and tracks faces in video data, using a separate thread
struct TrackThread {
    TrackThread(void);
    ~TrackThread(void);

    struct Output {
        Rect2d bbox;
        bool tracking{false};
    };

    // If the tracking thread is busy, it does nothing. Otherwise, it pushes a
    // new frame to the thread and refreshes "out" with the new tracking data
    // (saying if we are tracking something and the bounding box in the frame if
    // that is the case).
    void process(const Mat& in, Output& out);

private:
    mutex dataMtx;
    condition_variable frameCondition;
    Mat frame;
    Output output;
    bool debug, finish;
    // Keep this last as it uses the other members
    thread processThread;

    void threadMethod(void);
    Ptr<Tracker> createTracker(void);
};

TrackThread::TrackThread(void) :
    debug{getenv("FACETRACKER_DEBUG") ? true : false},
    finish{false},
    processThread{&TrackThread::threadMethod, this}
{
}

TrackThread::~TrackThread(void)
{
    {
        std::unique_lock<mutex> lock(dataMtx);
        finish = true;
        frameCondition.notify_one();
    }

    processThread.join();
}

Ptr<Tracker> TrackThread::createTracker(void)
{
    // List of tracker types in OpenCV 3.2
    string trackerTypes[6] = {"BOOSTING", "MIL", "KCF", "TLD",
                              "MEDIANFLOW", "GOTURN"};
    // Create a tracker and select type by choosing indicies
    //string trackerType = trackerTypes[0]; // BOOSTING -> always follows something...
    //string trackerType = trackerTypes[1]; // MIL -> always follows something...
    //string trackerType = trackerTypes[2]; // KCF -> always follows something...
    //string trackerType = trackerTypes[3]; // TLD -> bit slow
    string trackerType = trackerTypes[4]; // MEDIANFLOW -> Best trade-off atm
    //string trackerType = trackerTypes[5]; // GOTURN -> needs file, failing atm

    Ptr<Tracker> tracker;

    #if (CV_MAJOR_VERSION <= 3 && CV_MINOR_VERSION <= 2)
        tracker = Tracker::create(trackerType);
    #else
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
    #endif

    return tracker;
}

void TrackThread::threadMethod(void)
{
    bool tracking = false;
    Rect2d bbox;
    Ptr<Tracker> tracker;
    FaceDetector faceDetector;

    while (true) {
        {
            std::unique_lock<mutex> lock(dataMtx);
            frameCondition.wait(lock);

            if (finish)
                break;

            if (tracking)
                tracking = tracker->update(frame, bbox);

            if (!tracking) {
                tracking = faceDetector.detectFace(frame, bbox);
                if (tracking) {
                    // At least for KFC, we need to re-create the tracker when
                    // the tracked object changes. It looks like a repeated call
                    // to init does not fully clean the state and the
                    // performance of the tracker is greatly affected.
                    tracker = createTracker();
                    tracker->init(frame, bbox);
                }
            }

            output.bbox = bbox;
            output.tracking = tracking;

            if (debug)
                cout << "Size is " << frame.cols << " x " << frame.rows
                     << " . Tracking: " << tracking << '\n';
        }
    }
}

void TrackThread::process(const Mat& in, TrackThread::Output& out)
{
    static const double scale_f = 2.;

    std::unique_lock<mutex> lock(dataMtx, defer_lock_t());
    if (lock.try_lock()) {
        // Take latest track result
        out.tracking = output.tracking;
        out.bbox = Rect2d(scale_f*output.bbox.x,
                          scale_f*output.bbox.y,
                          scale_f*output.bbox.width,
                          scale_f*output.bbox.height);
        // Makes a copy to the shared frame
        resize(in, frame, Size(), 1/scale_f, 1/scale_f);
        frameCondition.notify_one();
    }
}

int main(int argc, char **argv)
{
    static const char *windowTitle = "Tracking";
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

    if (!video.isOpened()) {
        cout << "Could not open video source\n";
        return 1;
    }

    namedWindow(windowTitle, WINDOW_NORMAL);
    //setWindowProperty(windowTitle, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

    TrackThread tt;
    Mat in;
    TrackThread::Output out;
    while (video.read(in))
    {
        tt.process(in, out);

        if (out.tracking)
            rectangle(in, out.bbox, Scalar(255, 0, 0), 8, 1);

        imshow(windowTitle, in);

        // Exit if ESC pressed
        if (waitKey(1) == 27)
            break;
    }
}
