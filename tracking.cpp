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

struct TrackThread {
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

    bool detectFace(const Mat& frame, Rect2d& bbox);
};

bool TrackThread::detectFace(const Mat& frame, Rect2d& bbox)
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

    return true;
}

void TrackThread::operator()()
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
            std::unique_lock<mutex> lock(dataMtx);
            input.frameCondition.wait(lock);

            if (tracking)
                tracking = tracker->update(input.frame, bbox);

            if (!tracking) {
                tracking = detectFace(input.frame, bbox);
                if (tracking)
                    tracker->init(input.frame, bbox);
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
                if (tt.output.tracking) {
                    Rect2d bbox(scale_f*tt.output.bbox.x,
                                scale_f*tt.output.bbox.y,
                                scale_f*tt.output.bbox.width,
                                scale_f*tt.output.bbox.height);
                    rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
                }
                // Makes a copy to the shared frame
                resize(frame, tt.input.frame, cv::Size(), 1/scale_f, 1/scale_f);
                tt.input.frameCondition.notify_one();
            }
        }

        imshow("Tracking", frame);

        key = waitKey(1);
    }

    detectAndTrack.join();
}
