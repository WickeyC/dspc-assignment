#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono> // For measuring time
#include <pthread.h>
#include <iomanip>

//For suppressing info logs in debug mode
#include <opencv2/core/utils/logger.hpp>

using namespace cv;

struct GaussianThreadData {
    Mat* source;
    Mat* dst;
    cv::Mat* kernel2D;
    int startRow;
    int endRow;
};

struct CannyThreadArgs {
    cv::Mat* grad_x;
    cv::Mat* grad_y;
    cv::Mat* gradient;
    cv::Mat* dst;
    int startRow;
    int endRow;
    double low_thresh;
    double high_thresh;
};

struct RegionThreadData {
    Mat* source;
    Mat* mask;
    Mat* maskedImage;
    int startRow;
    int endRow;
};

struct HoughThreadData {
    cv::Mat image;
    int startIdx;
    int endIdx;
    std::vector<cv::Point> nzloc;
    float rho;
    float theta;
    int threshold;
    int lineLength;
    int lineGap;
    std::vector<cv::Vec4i> lines;
};

#define M_PI 3.14159265358979323846
#define NTHREADS 4

template <typename T, typename X>
X multiplyAndSum(std::vector<T> A, std::vector<T> B)
{
    X sum;
    std::vector<T> temp;
    for (int i = 0; i < A.size(); i++)
    {
        temp.push_back(A[i] * B[i]);
    }
    sum = std::accumulate(temp.begin(), temp.end(), 0.0);

    return sum;
}

template <typename T, typename X>
std::vector< X > estimateCoefficients(std::vector<T> A, std::vector<T> B)
{
    // Sample size
    int N = A.size();

    // Calculate mean of X and Y
    X meanA = std::accumulate(A.begin(), A.end(), 0.0) / A.size();
    X meanB = std::accumulate(B.begin(), B.end(), 0.0) / B.size();

    // Calculating cross-deviation and deviation about x
    X SSxy = multiplyAndSum<T, T>(A, B) - (N * meanA * meanB);
    X SSxx = multiplyAndSum<T, T>(A, A) - (N * meanA * meanA);

    // Calculating regression coefficients
    X slopeB1 = SSxy / SSxx;
    X interceptB0 = meanB - (slopeB1 * meanA);

    // Return vector, insert slope first and then intercept
    std::vector< X > coef;
    coef.push_back(slopeB1);
    coef.push_back(interceptB0);
    return coef;
}

bool isDayTime(Mat source)
{
    /* We've noticed that, in general, daytime images/videos require different color
    filters than nighttime images/videos. For example, in darker light it is better
    to add a gray color filter in addition to the white and yellow one */

    Scalar s = mean(source); // Mean pixel values 

    /* We chose these cut off values by looking at the mean pixel values of multiple
    daytime and nighttime images */
    if (s[0] < 30 || s[1] < 33 && s[2] < 30)
    {
        return false;
    }

    return true;
}

Mat filterColors(Mat source, bool isDayTime)
{
    Mat hsv, whiteMask, whiteImage, yellowMask, yellowImage, whiteYellow;
    auto startTime = std::chrono::high_resolution_clock::now(); // Record the start time
    // White mask
    std::vector< int > lowerWhite = { 130, 130, 130 };
    std::vector< int > upperWhite = { 255, 255, 255 };
    inRange(source, lowerWhite, upperWhite, whiteMask);
    bitwise_and(source, source, whiteImage, whiteMask);

    // Yellow mask
    cvtColor(source, hsv, COLOR_BGR2HSV);
    std::vector< int > lowerYellow = { 20, 100, 110 };
    std::vector< int > upperYellow = { 30, 180, 240 };
    inRange(hsv, lowerYellow, upperYellow, yellowMask);
    bitwise_and(source, source, yellowImage, yellowMask);

    // Blend yellow and white together
    addWeighted(whiteImage, 1., yellowImage, 1., 0., whiteYellow);

    // Add gray filter if image is not taken during the day
    if (isDayTime == false)
    {
        // Gray mask
        Mat grayMask, grayImage, grayAndWhite, dst;
        std::vector< int > lowerGray = { 80, 80, 80 };
        std::vector< int > upperGray = { 130, 130, 130 };
        inRange(source, lowerGray, upperGray, grayMask);
        bitwise_and(source, source, grayImage, grayMask);

        // Blend gray, yellow and white together and return the result
        addWeighted(grayImage, 1., whiteYellow, 1., 0., dst);
        auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
        auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds
        std::cout << "Colour Filter completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;
        return dst;
    }
    auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds
    std::cout << "Colour Filter completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    // Return white and yellow mask if image is taken during the day
    return whiteYellow;
}

Mat applyGrayscale(Mat source)
{
    Mat dst;

    auto startTime = std::chrono::high_resolution_clock::now(); // Record the start time
    cvtColor(source, dst, COLOR_BGR2GRAY);
    auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds
    std::cout << "Gray Scaling completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return dst;
}

void* gaussianBlurThread(void* arg) {
    GaussianThreadData* data = (GaussianThreadData*)arg;
    Mat& source = *(data->source);
    Mat& dst = *(data->dst);
    cv::Mat& kernel2D = *(data->kernel2D);

    int kernelSize = kernel2D.rows;

    for (int y = data->startRow; y < data->endRow; y++) {
        for (int x = 0; x < source.cols; x++) {
            double weightedSum = 0.0;
            for (int ky = -kernelSize / 2; ky <= kernelSize / 2; ky++) {
                for (int kx = -kernelSize / 2; kx <= kernelSize / 2; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;

                    if (nx >= 0 && nx < source.cols && ny >= 0 && ny < source.rows) {
                        double weight = kernel2D.at<double>(ky + kernelSize / 2, kx + kernelSize / 2);
                        weightedSum += source.at<uchar>(ny, nx) * weight;
                    }
                }
            }
            dst.at<uchar>(y, x) = static_cast<uchar>(weightedSum);
        }
    }
    return nullptr;
}

Mat applyGaussianBlur(Mat source) {
    Mat dst(source.rows, source.cols, source.type());

    auto startTime = std::chrono::high_resolution_clock::now();

    int kernelSize = 3;
    double sigma = 0.0;
    if (sigma == 0.0) {
        sigma = 0.3 * ((kernelSize - 1) * 0.5 - 1) + 0.8;
    }
    cv::Mat kernel = cv::getGaussianKernel(kernelSize, sigma, CV_64F);
    cv::Mat kernel2D = kernel * kernel.t();
    kernel2D /= cv::sum(kernel2D)[0];

    int numThreads = NTHREADS;
    pthread_t threads[NTHREADS];
    GaussianThreadData threadData[NTHREADS];
    int rowsPerThread = source.rows / numThreads;

    for (int i = 0; i < numThreads; i++) {
        threadData[i].source = &source;
        threadData[i].dst = &dst;
        threadData[i].kernel2D = &kernel2D;
        threadData[i].startRow = i * rowsPerThread;
        threadData[i].endRow = (i == numThreads - 1) ? source.rows : (i + 1) * rowsPerThread;
        pthread_create(&threads[i], nullptr, gaussianBlurThread, &threadData[i]);
    }

    for (int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], nullptr);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0;
    std::cout << "Gaussian Blur completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return dst;
}

void* cannyThread(void* args) {
    CannyThreadArgs* targs = (CannyThreadArgs*)args;
    for (int i = targs->startRow; i < targs->endRow; i++) {
        // ... (rest of the code inside the double for loop)
    }
    return NULL;
}

Mat applyCanny(Mat source, double low_thresh = 50, double high_thresh = 100, int aperture_size = 3, bool L2gradient = false) {
    Mat dst;
    auto startTime = std::chrono::high_resolution_clock::now(); // Record the start time

    Mat src = source.clone(); // Clone the source image to avoid modifying it
    cv::Size size = src.size();
    cv::Mat gradient;

    // Calculate gradients (Sobel operators)
    cv::Mat grad_x, grad_y;
    cv::Sobel(src, grad_x, CV_16S, 1, 0, aperture_size);
    cv::Sobel(src, grad_y, CV_16S, 0, 1, aperture_size);

    // Compute gradient magnitude and direction
    cv::Mat abs_grad_x, abs_grad_y;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gradient);

    // Non-maximum suppression
    dst = cv::Mat::zeros(size, CV_8U);

    // Create and initialize thread arguments
    int num_threads = NTHREADS;
    pthread_t threads[NTHREADS];
    CannyThreadArgs threadArgs[NTHREADS];
    int numRowsPerThread = size.height / num_threads;

    for (int t = 0; t < num_threads; t++) {
        threadArgs[t].grad_x = &grad_x;
        threadArgs[t].grad_y = &grad_y;
        threadArgs[t].gradient = &gradient;
        threadArgs[t].dst = &dst;
        threadArgs[t].startRow = t * numRowsPerThread;
        threadArgs[t].endRow = (t == num_threads - 1) ? size.height : (t + 1) * numRowsPerThread;
        threadArgs[t].low_thresh = low_thresh;
        threadArgs[t].high_thresh = high_thresh;
        pthread_create(&threads[t], NULL, cannyThread, &threadArgs[t]);
    }

    // Join threads
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds
    std::cout << "Canny Edge Detection completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return dst;
}

void* maskingThread(void* arg) {
    RegionThreadData* data = (RegionThreadData*)arg;

    for (int i = data->startRow; i < data->endRow; i++) {
        for (int j = 0; j < data->source->cols; j++) {
            if (data->mask->at<uchar>(i, j) != 0) {
                data->maskedImage->at<uchar>(i, j) = data->source->at<uchar>(i, j);
            }
        }
    }
    return nullptr;
}

Mat RegionOfInterest(Mat source) {
    // Trapezoid parameters and mask creation
    float trapezoidBottomWidth = 1.0;
    float trapezoidTopWidth = 0.07;
    float trapezoidHeight = 0.5;
    float smallBottomWidth = 0.45;
    float smallTopWidth = 0.3;
    float smallHeight = 1.0;
    float bar = 0.97;

    std::vector<Point> pts;

    auto startTime = std::chrono::high_resolution_clock::now(); // Record the start time

    // Large trapezoid
    pts.push_back(Point((source.cols * (1 - trapezoidBottomWidth)) / 2, source.rows * bar));
    pts.push_back(Point((source.cols * (1 - trapezoidTopWidth)) / 2, source.rows - source.rows * trapezoidHeight));
    pts.push_back(Point(source.cols - (source.cols * (1 - trapezoidTopWidth)) / 2, source.rows - source.rows * trapezoidHeight));
    pts.push_back(Point(source.cols - (source.cols * (1 - trapezoidBottomWidth)) / 2, source.rows * bar));

    // Small trapezoid
    pts.push_back(Point((source.cols * (1 - trapezoidBottomWidth + smallBottomWidth)) / 2, source.rows * bar));
    pts.push_back(Point((source.cols * (1 - trapezoidTopWidth * smallTopWidth)) / 2, source.rows - source.rows * trapezoidHeight * smallHeight));
    pts.push_back(Point(source.cols - (source.cols * (1 - trapezoidTopWidth * smallTopWidth)) / 2, source.rows - source.rows * trapezoidHeight * smallHeight));
    pts.push_back(Point(source.cols - (source.cols * (1 - trapezoidBottomWidth + smallBottomWidth)) / 2, source.rows * bar));

    Mat mask = Mat::zeros(source.size(), source.type());
    fillPoly(mask, pts, Scalar(255, 255, 255));

    // Parallel masking
    Mat maskedImage = Mat::zeros(source.size(), source.type());

    int numThreads = NTHREADS;
    pthread_t threads[NTHREADS];
    RegionThreadData threadData[NTHREADS];
    int rowsPerThread = source.rows / numThreads;

    for (int i = 0; i < numThreads; i++) {
        threadData[i].source = &source;
        threadData[i].mask = &mask;
        threadData[i].maskedImage = &maskedImage;
        threadData[i].startRow = i * rowsPerThread;
        threadData[i].endRow = (i == numThreads - 1) ? source.rows : (i + 1) * rowsPerThread;
        pthread_create(&threads[i], nullptr, maskingThread, &threadData[i]);
    }

    for (int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], nullptr);
    }

    auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds
    std::cout << "Region Masking completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return maskedImage;
}

// START: for Hough Transform Probabilistics
void* HoughWorker(void* arg) {
    HoughThreadData* data = (HoughThreadData*)arg;

    cv::Point pt;
    cv::RNG rng((uint64)-1);
    int width = data->image.cols;
    int height = data->image.rows;
    float irho = 1 / data->rho;

    // Other initializations from HoughLinesProbabilistic
    int numangle = cvFloor((CV_PI - 0.0) / data->theta) + 1;
    if (numangle > 1 && fabs(CV_PI - (numangle - 1) * data->theta) < data->theta / 2)
        --numangle;
    int numrho = cvRound(((width + height) * 2 + 1) / data->rho);
    cv::Mat accum = cv::Mat::zeros(numangle, numrho, CV_32SC1);
    std::vector<float> trigtab(numangle * 2);

    for (int n = 0; n < numangle; n++) {
        trigtab[n * 2] = (float)(cos((double)n * data->theta) * irho);
        trigtab[n * 2 + 1] = (float)(sin((double)n * data->theta) * irho);
    }

    const float* ttab = &trigtab[0];
    uchar* mdata0 = data->image.ptr();  // Adjusted based on the passed image data

    int count = data->endIdx - data->startIdx;

    // Stage 2: Process all the points in the specified range
    for (int idx = data->startIdx; idx < data->endIdx; idx++) {
        int max_val = data->threshold - 1, max_n = 0;
        cv::Point point = data->nzloc[idx];
        cv::Point line_end[2];
        float a, b;
        int* adata = accum.ptr<int>();
        int i = point.y, j = point.x, k, x0, y0, dx0, dy0, xflag;
        int good_line;
        const int shift = 16;

        if (!mdata0[i * width + j])
            continue;

        for (int n = 0; n < numangle; n++, adata += numrho) {
            int r = cvRound(j * ttab[n * 2] + i * ttab[n * 2 + 1]);
            r += (numrho - 1) / 2;
            int val = ++adata[r];
            if (max_val < val) {
                max_val = val;
                max_n = n;
            }
        }

        if (max_val < data->threshold)
            continue;

        a = -ttab[max_n * 2 + 1];
        b = ttab[max_n * 2];
        x0 = j;
        y0 = i;
        if (fabs(a) > fabs(b)) {
            xflag = 1;
            dx0 = a > 0 ? 1 : -1;
            dy0 = cvRound(b * (1 << shift) / fabs(a));
            y0 = (y0 << shift) + (1 << (shift - 1));
        }
        else {
            xflag = 0;
            dy0 = b > 0 ? 1 : -1;
            dx0 = cvRound(a * (1 << shift) / fabs(b));
            x0 = (x0 << shift) + (1 << (shift - 1));
        }

        for (k = 0; k < 2; k++) {
            int gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;

            if (k > 0)
                dx = -dx, dy = -dy;

            for (;; x += dx, y += dy) {
                uchar* mdata;
                int i1, j1;

                if (xflag) {
                    j1 = x;
                    i1 = y >> shift;
                }
                else {
                    j1 = x >> shift;
                    i1 = y;
                }

                if (j1 < 0 || j1 >= width || i1 < 0 || i1 >= height)
                    break;

                mdata = mdata0 + i1 * width + j1;

                if (*mdata) {
                    gap = 0;
                    line_end[k].y = i1;
                    line_end[k].x = j1;
                }
                else if (++gap > data->lineGap) {
                    break;
                }
            }
        }

        good_line = std::abs(line_end[1].x - line_end[0].x) >= data->lineLength || std::abs(line_end[1].y - line_end[0].y) >= data->lineLength;

        for (k = 0; k < 2; k++) {
            int x = x0, y = y0, dx = dx0, dy = dy0;

            if (k > 0)
                dx = -dx, dy = -dy;

            for (;; x += dx, y += dy) {
                uchar* mdata;
                int i1, j1;

                if (xflag) {
                    j1 = x;
                    i1 = y >> shift;
                }
                else {
                    j1 = x >> shift;
                    i1 = y;
                }

                mdata = mdata0 + i1 * width + j1;

                if (*mdata) {
                    if (good_line) {
                        adata = accum.ptr<int>();
                        for (int n = 0; n < numangle; n++, adata += numrho) {
                            int r = cvRound(j1 * ttab[n * 2] + i1 * ttab[n * 2 + 1]);
                            r += (numrho - 1) / 2;
                            adata[r]--;
                        }
                    }
                    *mdata = 0;
                }

                if (i1 == line_end[k].y && j1 == line_end[k].x)
                    break;
            }
        }

        if (good_line) {
            cv::Vec4i lr(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
            data->lines.push_back(lr);
            if ((int)data->lines.size() >= data->lineGap)
                return nullptr;
        }
    }
    return nullptr;
}

void HoughLinesProbabilistic(cv::Mat& image,
    float rho, float theta, int threshold,
    int lineLength, int lineGap,
    std::vector<cv::Vec4i>& lines, int linesMax) {
    cv::Point pt;
    float irho = 1 / rho;
    cv::RNG rng((uint64)-1);

    CV_Assert(image.type() == CV_8UC1);

    int width = image.cols;
    int height = image.rows;

    //compute numangle
    int numangle = cvFloor((CV_PI - 0.0) / theta) + 1;
    // If the distance between the first angle and the last angle is
    // approximately equal to pi, then the last angle will be removed
    // in order to prevent a line to be detected twice.
    if (numangle > 1 && fabs(CV_PI - (numangle - 1) * theta) < theta / 2)
        --numangle;

    int numrho = cvRound(((width + height) * 2 + 1) / rho);

    cv::Mat accum = cv::Mat::zeros(numangle, numrho, CV_32SC1);
    cv::Mat mask(height, width, CV_8UC1);
    std::vector<float> trigtab(numangle * 2);

    for (int n = 0; n < numangle; n++)
    {
        trigtab[n * 2] = (float)(cos((double)n * theta) * irho);
        trigtab[n * 2 + 1] = (float)(sin((double)n * theta) * irho);
    }
    const float* ttab = &trigtab[0];
    uchar* mdata0 = mask.ptr();
    std::vector<cv::Point> nzloc;

    // Stage 1: Collect non-zero image points (same as original)

    int numThreads = NTHREADS;
    pthread_t threads[NTHREADS];
    HoughThreadData threadData[NTHREADS];

    int chunkSize = nzloc.size() / numThreads;

    for (int t = 0; t < NTHREADS; t++) {
        threadData[t].image = image;
        threadData[t].startIdx = t * chunkSize;
        threadData[t].endIdx = (t == NTHREADS - 1) ? nzloc.size() : (t + 1) * chunkSize;
        threadData[t].nzloc = nzloc;
        threadData[t].rho = rho;
        threadData[t].theta = theta;
        threadData[t].threshold = threshold;
        threadData[t].lineLength = lineLength;
        threadData[t].lineGap = lineGap;
        pthread_create(&threads[t], NULL, HoughWorker, &threadData[t]);
    }

    for (int t = 0; t < NTHREADS; t++) {
        pthread_join(threads[t], NULL);
        lines.insert(lines.end(), threadData[t].lines.begin(), threadData[t].lines.end());
    }
}

void customHoughLinesP(cv::Mat canny, std::vector<cv::Vec4i>& linesP,
    double rho, double theta, int thresh,
    double minLineLength, double maxGapLength)
{
    linesP.clear(); // Clear the output vector to ensure it's empty.

    std::vector<cv::Vec4i> lines;
    HoughLinesProbabilistic(canny, (float)rho, (float)theta, thresh, cvRound(minLineLength), cvRound(maxGapLength), lines, INT_MAX);

    linesP = lines; // Copy the computed lines to the output vector.
}

std::vector<Vec4i> houghLines(Mat canny, Mat source, bool drawHough)
{
    double rho = 2; // Distance resolution in pixels of the Hough grid
    double theta = 1 * M_PI / 180; // Angular resolution in radians of Hough grid
    int thresh = 15; // Minimum number of votes (intersections in Hough grid cell)
    double minLineLength = 10; // Minimum number of pixels making up a line
    double maxGapLength = 20; // Max gap in pixels btwn connectable line segments

    std::vector<Vec4i> linesP; // Will hold the results of the detection

    auto startTime = std::chrono::high_resolution_clock::now(); // Record the start time

    customHoughLinesP(canny, linesP, rho, theta, thresh, minLineLength, maxGapLength);

    if (drawHough == true)
    {
        for (size_t i = 0; i < linesP.size(); i++)
        {
            Vec4i l = linesP[i];
            line(source, Point(l[0], l[1]), Point(l[2], l[3]),
                Scalar(0, 0, 255), 3, LINE_AA);
        }

        namedWindow("Hough Lines Detected", WINDOW_NORMAL);  // Create a resizable window
        resizeWindow("Hough Lines Detected", 527, 392);      // Set the desired size

        imshow("Hough Lines Detected", source);
    }

    auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds
    std::cout << "Hough Transform completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return linesP;
}
// END: for Hough Transform Probabilistics

Mat drawLanes(Mat source, std::vector<Vec4i> lines)
{
    // Stop if there are no lines, just return original image without lines
    if (lines.size() == 0)
    {
        return source;
    }

    // Set drawing lanes to true
    bool drawRightLane = true;
    bool drawLeftLane = true;

    // Find lines with a slope higher than the slope threshold
    float slopeThreshold = 0.5;
    std::vector< float > slopes;
    std::vector< Vec4i > goodLines;

    auto startTime = std::chrono::high_resolution_clock::now(); // Record the start time
    for (int i = 0; i < lines.size(); i++)
    {
        /* Each line is represented by a 4-element vector (x_1, y_1, x_2, y_2),
        where (x_1,y_1) is the line's starting point and (x_2, y_2) is the ending point */
        Vec4i l = lines[i];
        double slope;

        // Calculate slope
        if (l[2] - l[0] == 0 || l[0] == 0) // Avoid division by zero
        {
            slope = 999; // Basically infinte slope
        }
        else
        {
            slope = (l[3] - l[1]) / (l[2] / l[0]);
        }
        if (abs(slope) > 0.5)
        {
            slopes.push_back(slope);
            goodLines.push_back(l);
        }
    }

    /* Split the good lines into two categories: right and left
    The right lines have a positive slope and the left lines have a negative slope */
    std::vector< Vec4i > rightLines;
    std::vector< Vec4i > leftLines;
    int imgCenter = source.cols / 2;

    for (int i = 0; i < slopes.size(); i++)
    {
        if (slopes[i] > 0 && goodLines[i][0] > imgCenter && goodLines[i][2] > imgCenter)
        {
            rightLines.push_back(goodLines[i]);
        }
        if (slopes[i] < 0 && goodLines[i][0] < imgCenter && goodLines[i][2] < imgCenter)
        {
            leftLines.push_back(goodLines[i]);
        }
    }

    /* Now that we've isolated the right lane lines from the left lane lines,
    it is time to form two lane lines out of all the lines we've detected.
    A line is defined as 2 points: a starting point and an ending point.
    So up to this point the right and left lane basically consist of multiple hough lines.
    Our goal at this step is to use linear regression to find the two best fitting lines:
    one through the points at the left side to form the left lane
    and one through the points at the right side to form the right lane */

    // We start with the right side points
    std::vector< int > rightLinesX;
    std::vector< int > rightLinesY;
    double rightB1, rightB0; // Slope and intercept

    for (int i = 0; i < rightLines.size(); i++)
    {
        rightLinesX.push_back(rightLines[i][0]); // X of starting point of line
        rightLinesX.push_back(rightLines[i][2]); // X of ending point of line
        rightLinesY.push_back(rightLines[i][1]); // Y of starting point of line
        rightLinesY.push_back(rightLines[i][3]); // Y of ending point of line
    }

    if (rightLinesX.size() > 0)
    {
        std::vector< double > coefRight = estimateCoefficients<int, double>(rightLinesX, rightLinesY); // y = b1x + b0
        rightB1 = coefRight[0];
        rightB0 = coefRight[1];
    }
    else
    {
        rightB1 = 1;
        rightB0 = 1;
        drawRightLane = false;
    }

    // Now the points at the left side
    std::vector< int > leftLinesX;
    std::vector< int > leftLinesY;
    double leftB1, leftB0; // Slope and intercept

    for (int i = 0; i < leftLines.size(); i++)
    {
        leftLinesX.push_back(leftLines[i][0]); // X of starting point of line
        leftLinesX.push_back(leftLines[i][2]); // X of ending point of line
        leftLinesY.push_back(leftLines[i][1]); // Y of starting point of line
        leftLinesY.push_back(leftLines[i][3]); // Y of ending point of line
    }

    if (leftLinesX.size() > 0)
    {
        std::vector< double > coefLeft = estimateCoefficients<int, double>(leftLinesX, leftLinesY); // y = b1x + b0
        leftB1 = coefLeft[0];
        leftB0 = coefLeft[1];
    }
    else
    {
        leftB1 = 1;
        leftB0 = 1;
        drawLeftLane = false;
    }

    /* Now we need to find the two points for the right and left lane:
    starting points and ending points */

    int y1 = source.rows; // Y coordinate of starting point of both the left and right lane

    /* 0.5 = trapezoidHeight (see RegionOfInterest), we set the y coordinate of the ending point
    below the trapezoid height (0.4) to draw shorter lanes. I think that looks nicer. */

    int y2 = source.rows * (1 - 0.4); // Y coordinate of ending point of both the left and right lane

    // y = b1x + b0 --> x = (y - b0) / b1
    int rightX1 = (y1 - rightB0) / rightB1; // X coordinate of starting point of right lane
    int rightX2 = (y2 - rightB0) / rightB1; // X coordinate of ending point of right lane

    int leftX1 = (y1 - leftB0) / leftB1; // X coordinate of starting point of left lane
    int leftX2 = (y2 - leftB0) / leftB1; // X coordinate of ending point of left lane

    /* If the ending point of the right lane is on the left side of the left lane (or vice versa),
    return source image without drawings, because this should not be happening in real life. */
    if (rightX2 < leftX2 || leftX2 > rightX2)
    {
        return source;
    }

    // Create the mask
    Mat mask = Mat::zeros(source.size(), source.type());

    // Draw lines and fill poly made up of the four points described above if both bools are true
    Mat dst; // Holds blended image
    if (drawRightLane == true && drawLeftLane == true)
    {
        line(source, Point(rightX1, y1), Point(rightX2, y2), Scalar(255, 0, 0), 7);
        line(source, Point(leftX1, y1), Point(leftX2, y2), Scalar(255, 0, 0), 7);

        Point pts[4] = {
        Point(leftX1, y1), // Starting point left lane
        Point(leftX2, y2), // Ending point left lane
        Point(rightX2, y2), // Ending point right lane
        Point(rightX1, y1) // Starting point right lane
        };

        fillConvexPoly(mask, pts, 4, Scalar(235, 229, 52)); // Color is light blue

        // Blend the mask and source image together
        addWeighted(source, 0.9, mask, 0.3, 0.0, dst);

        // Return blended image
        auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
        auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds
        std::cout << "Lane Drawing completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;
        return dst;
    }

    auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds
    std::cout << "Lane Drawing completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return source; // Return source if drawing lanes did not happen
}

void drawLane(Mat image)
{
    // Record the start time
    auto startTime = std::chrono::high_resolution_clock::now();
    // Determine if video is taken during daytime or not
    //bool isDay = isDayTime(image);

    // Filter image 
    //Mat filteredIMG = filterColors(image, isDay);

    // Apply grayscale
    Mat gray = applyGrayscale(image);

    // Apply Gaussian blur
    Mat gBlur = applyGaussianBlur(gray);

    // Find edges
    Mat edges = applyCanny(gBlur);

    // Create mask (Region of Interest)
    Mat maskedIMG = RegionOfInterest(edges);

    // Detect straight lines and draw the lanes if possible
    std::vector<cv::Vec4i> linesP = houghLines(maskedIMG, image.clone(), true);
    Mat lanes = drawLanes(image, linesP);
    // Record the end time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds
    std::cout << "===============================================" << std::endl;
    std::cout << "FRAME PROCESSING COMPLETED IN " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;
    std::cout << "===============================================" << std::endl;

    // Create a window with a fixed size
    namedWindow("Lane Detection App", WINDOW_NORMAL);
    resizeWindow("Lane Detection App", 527, 392);

    // Display the image in the named window
    imshow("Lane Detection App", lanes);
}

int main(int argc, char* argv[])
{
    // Control OpenCV Log in Debug Mode (For a clearer console)
    // Suppress all Log
    //utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
    // Only shows Error level log
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

    if (argc != 2) {
        std::cout << "Usage: ./exe path-to-video-or-image" << std::endl;
        return -1;
    }

    std::string inputPath = argv[1];

    // Load video or image based on file extension
    bool isVideo = inputPath.find(".mp4") != std::string::npos;

    if (isVideo) {
        // Initialize video capture
        VideoCapture cap(inputPath);

        // Check if video can be opened
        if (!cap.isOpened())
        {
            std::cout << "Failed to open videofile!" << std::endl;
            return -1;
        }
        Mat inputFrame;
        cap >> inputFrame;
        if (inputFrame.empty()) {
            std::cout << "Empty frame from the video!" << std::endl;
            return -1;
        }

        // Display video with lanes drawn
        while (true) {
            Mat frame;
            cap >> frame;

            // Stop if frame is empty (end of video)
            if (frame.empty())
            {
                break;
            }
            drawLane(frame);
            if (waitKey(1) == 27) break;
        }
    }
    else if (inputPath.find(".png") != std::string::npos ||
        inputPath.find(".jpg") != std::string::npos ||
        inputPath.find(".jpeg") != std::string::npos) {
        // Load the image
        Mat image = imread(inputPath);
        if (image.empty()) {
            std::cout << "Failed to load image!" << std::endl;
            return -1;
        }

        drawLane(image);
        waitKey(0);
    }
    else {
        std::cout << "Unsupported file format!" << std::endl;
        return -1;
    }

    return 0;
}