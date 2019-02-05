This is an example of face detection and tracking using OpenCV.  The
program runs a face detector until there is a positive. At that point it
tracks the face until the tracker fails, and then it comes back to using
the face detector, in a loop.

The program has separate threads for video presentation on the display
and frame processing using OpenCV, to avoid video stagnation while doing
computer vision tasks, which can easily happen in low end CPUs.  The
video presentation thread sends frames to the computer vision thread,
but only if the later is not busy at that moment - that is, some frames
will be shown on the screen but not processed if there are not enough
computing resources. This makes the video fluid even in low end
processors.
