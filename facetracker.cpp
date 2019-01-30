/*******************************************************************************
 * Author     : Alfonso Sanchez-Beato, based on example from Manu BN
 * Description: Detect face using OpenCV's Haar cascade and track it until lost.
 *  At that point, try to detect face again and track in loop.
 ******************************************************************************/
#include <stdlib.h>

#include <mutex>
#include <thread>
#include <condition_variable>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#define HAAR_DATA_DIR "/usr/share/opencv/haarcascades/"

using namespace cv;
using namespace std;

struct TrackThread {
    TrackThread(void);

    mutex dataMtx;
    struct Input {
        Input(void) : finish{false} {}
        condition_variable frameCondition;
        Mat frame;
        bool finish;
    } input;
    struct Output {
        Output(void) : tracking{false} {}
        Rect2d bbox;
        bool tracking;
    } output;

    void operator()(void);

private:
    CascadeClassifier faceCascade;

    Ptr<Tracker> createTracker(void);
    bool detectFace(const Mat& frame, Rect2d& bbox);
};

TrackThread::TrackThread(void)
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

bool TrackThread::detectFace(const Mat& frame, Rect2d& bbox)
{
    // Detect face using Haar Cascade classifier
    vector<Rect> f;
    // See http://www.emgu.com/wiki/files/1.5.0.0/Help/html/e2278977-87ea-8fa9-b78e-0e52cfe6406a.htm
    // for flag description. It might be wortwhile to play a bit with the
    // different parameters.
    faceCascade.detectMultiScale(frame, f, 1.1, 2, CASCADE_SCALE_IMAGE);
    if (f.size() == 0)
        return false;

    cout << "Detected " << f.size() << " faces \n";
    // Get only one face for the moment
    bbox = Rect2d(f[0].x, f[0].y, f[0].width, f[0].height);

    return true;
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

    #if (CV_MAJOR_VERSION <= 3 && CV_MINOR_VERSION < 3)
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

    return tracker;
}

void TrackThread::operator()(void)
{
    bool tracking = false;
    Rect2d bbox;
    Ptr<Tracker> tracker;

    while (true) {
        {
            std::unique_lock<mutex> lock(dataMtx);
            input.frameCondition.wait(lock);

            if (tracking)
                tracking = tracker->update(input.frame, bbox);

            if (!tracking) {
                tracking = detectFace(input.frame, bbox);
                if (tracking) {
                    // At least for KFC, we need to re-create the tracker when
                    // the tracked object changes. It looks like a repeated call
                    // to init does not fully clean the state and the
                    // performance of the tracker is greatly affected.
                    tracker = createTracker();
                    tracker->init(input.frame, bbox);
                }
            }

            output.bbox = bbox;
            output.tracking = tracking;

            cout << "Size is " << input.frame.cols << " x " << input.frame.rows
                 << " . Tracking: " << tracking << '\n';

            if (input.finish)
                break;
        }
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
    if (!video.isOpened()) {
        cout << "Could not read video file" << endl;
        return 1;
    }

    TrackThread tt;
    thread detectAndTrack{ref(tt)};

    double scale_f = 2.;
    Mat frame;
    int key = 0;
    bool tracking = false;
    Rect2d bbox;
    while (video.read(frame))
    {
        {
            std::unique_lock<mutex> lock(tt.dataMtx, defer_lock_t());
            if (lock.try_lock()) {
                // Exit if ESC pressed.
                if(key == 27) {
                    tt.input.finish = true;
                    tt.input.frameCondition.notify_one();
                    break;
                }
                // Take latest track result
                tracking = tt.output.tracking;
                bbox = Rect2d(scale_f*tt.output.bbox.x,
                              scale_f*tt.output.bbox.y,
                              scale_f*tt.output.bbox.width,
                              scale_f*tt.output.bbox.height);
                // Makes a copy to the shared frame
                resize(frame, tt.input.frame, cv::Size(), 1/scale_f, 1/scale_f);
                tt.input.frameCondition.notify_one();
            }
        }

        if (tracking)
            rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
        imshow("Tracking", frame);

        key = waitKey(1);
    }

    detectAndTrack.join();
}
