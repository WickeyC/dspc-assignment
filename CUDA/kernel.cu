#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono> // For measuring time

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//For suppressing info logs in debug mode
#include <opencv2/core/utils/logger.hpp>

using namespace cv;

#define M_PI 3.14159265358979323846

#define CUDA_CHECK(call) { \
    cudaError_t cudaError = call; \
    if (cudaError != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(cudaError) << " at Line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

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

__global__ void gaussianBlurCUDA(const uchar* src, uchar* dst, int width, int height, float* kernel, int kernelSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int offset = y * width + x;
        float sum = 0.0f;
        float sumWeights = 0.0f; // Variable to store the sum of kernel weights

        // Compute the sum of weights in the kernel
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                int offsetX = x + j - kernelSize / 2;
                int offsetY = y + i - kernelSize / 2;

                if (offsetX >= 0 && offsetX < width && offsetY >= 0 && offsetY < height)
                {
                    float weight = kernel[i * kernelSize + j];
                    sumWeights += weight;
                }
            }
        }

        // Apply convolution with normalization
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                int offsetX = x + j - kernelSize / 2;
                int offsetY = y + i - kernelSize / 2;

                if (offsetX >= 0 && offsetX < width && offsetY >= 0 && offsetY < height)
                {
                    int srcOffset = offsetY * width + offsetX;
                    float weight = kernel[i * kernelSize + j];
                    sum += static_cast<float>(src[srcOffset]) * weight;
                }
            }
        }

        // Normalize the result
        sum /= sumWeights;

        dst[offset] = static_cast<uchar>(sum);
    }
}

cv::Mat applyGaussianBlur(const cv::Mat& source, int kernelSize)
{
    auto startTime = std::chrono::high_resolution_clock::now(); // Record the start time

    int width = source.cols;
    int height = source.rows;
    int imageSize = width * height * sizeof(uchar);

    // Define a Gaussian kernel
    float gaussianKernel[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
    float* d_kernel;

    // Allocate memory for the kernel on the GPU
    cudaMalloc((void**)&d_kernel, sizeof(float) * kernelSize * kernelSize);
    cudaMemcpy(d_kernel, gaussianKernel, sizeof(float) * kernelSize * kernelSize, cudaMemcpyHostToDevice);

    uchar* d_src;
    uchar* d_dst;

    // Allocate memory for source and destination images on the GPU
    cudaMalloc((void**)&d_src, imageSize);
    cudaMalloc((void**)&d_dst, imageSize);

    // Copy the source image to the GPU
    cudaMemcpy(d_src, source.data, imageSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    gaussianBlurCUDA << <grid, block >> > (d_src, d_dst, width, height, d_kernel, kernelSize);

    cv::Mat result(height, width, CV_8U);
    cudaMemcpy(result.data, d_dst, imageSize, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_kernel);
    cudaFree(d_src);
    cudaFree(d_dst);

    auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds
    std::cout << "Gaussian Blur completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return result;
}

__global__ void cannyCUDA(const short* grad_x, const short* grad_y, uchar* dst, int width, int height, double low_thresh, double high_thresh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        int offset = y * width + x;

        int dx = grad_x[offset];
        int dy = grad_y[offset];

        float gradientValue = sqrt(static_cast<float>(dx * dx + dy * dy));

        uchar* pixel = dst + offset;

        if (gradientValue >= high_thresh)
        {
            *pixel = 255;
        }
        else if (gradientValue >= low_thresh && gradientValue < high_thresh)
        {
            if (grad_x[offset + 1] >= high_thresh ||
                grad_x[offset - 1] >= high_thresh ||
                grad_x[offset + width] >= high_thresh ||
                grad_x[offset - width] >= high_thresh ||
                grad_x[offset + width + 1] >= high_thresh ||
                grad_x[offset - width - 1] >= high_thresh ||
                grad_x[offset + width - 1] >= high_thresh ||
                grad_x[offset - width + 1] >= high_thresh ||
                grad_y[offset + 1] >= high_thresh ||
                grad_y[offset - 1] >= high_thresh ||
                grad_y[offset + width] >= high_thresh ||
                grad_y[offset - width] >= high_thresh ||
                grad_y[offset + width + 1] >= high_thresh ||
                grad_y[offset - width - 1] >= high_thresh ||
                grad_y[offset + width - 1] >= high_thresh ||
                grad_y[offset - width + 1] >= high_thresh)
            {
                *pixel = 255;
            }
        }
    }
}

cv::Mat applyCanny(const cv::Mat& source, double low_thresh = 50, double high_thresh = 100, int aperture_size = 3, bool L2gradient = false)
{
    cv::Mat dst;
    auto startTime = std::chrono::high_resolution_clock::now(); // Record the start time

    cv::Mat src = source.clone(); // Clone the source image to avoid modifying it
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

    dst = cv::Mat::zeros(size, CV_8U);

    // Allocate GPU memory and copy data
    short* d_grad_x;
    short* d_grad_y;
    uchar* d_dst;

    cudaMalloc((void**)&d_grad_x, size.width * size.height * sizeof(short));
    cudaMalloc((void**)&d_grad_y, size.width * size.height * sizeof(short));
    cudaMalloc((void**)&d_dst, size.width * size.height * sizeof(uchar));

    cudaMemcpy(d_grad_x, grad_x.ptr<short>(), size.width * size.height * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_y, grad_y.ptr<short>(), size.width * size.height * sizeof(short), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((size.width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);

    cannyCUDA << <grid, block >> > (d_grad_x, d_grad_y, d_dst, size.width, size.height, low_thresh, high_thresh);

    cudaMemcpy(dst.data, d_dst, size.width * size.height * sizeof(uchar), cudaMemcpyDeviceToHost);

    cudaFree(d_grad_x);
    cudaFree(d_grad_y);
    cudaFree(d_dst);

    auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds
    std::cout << "Canny Edge Detection completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return dst;
}

__global__ void bitwiseAndCUDA(const uchar* mask, const uchar* source, uchar* result, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int offset = y * width + x;
        result[offset] = (mask[offset] != 0) ? source[offset] : 0;
    }
}

cv::Mat RegionOfInterest(const cv::Mat& source)
{
    float trapezoidBottomWidth = 1.0; // Width of bottom edge of trapezoid, expressed as a percentage of image width
    float trapezoidTopWidth = 0.07;   // Width of top edge of trapezoid, expressed as a percentage of image width
    float trapezoidHeight = 0.5;      // Height of the trapezoid, expressed as a percentage of image height
    float bar = 0.97;                 // Parameter to control the trapezoid position

    std::vector<cv::Point> pts;

    auto startTime = std::chrono::high_resolution_clock::now(); // Record the start time

    // Define points for the large trapezoid
    pts.push_back(cv::Point((source.cols * (1 - trapezoidBottomWidth)) / 2, source.rows * bar)); // Bottom left
    pts.push_back(cv::Point((source.cols * (1 - trapezoidTopWidth)) / 2, source.rows - source.rows * trapezoidHeight)); // Top left
    pts.push_back(cv::Point(source.cols - (source.cols * (1 - trapezoidTopWidth)) / 2, source.rows - source.rows * trapezoidHeight)); // Top right
    pts.push_back(cv::Point(source.cols - (source.cols * (1 - trapezoidBottomWidth)) / 2, source.rows * bar)); // Bottom right

    // Define points for the small trapezoid
    float smallBottomWidth = 0.45;
    float smallTopWidth = 0.3;
    float smallHeight = 1.0;
    pts.push_back(cv::Point((source.cols * (1 - trapezoidBottomWidth + smallBottomWidth)) / 2, source.rows * bar)); // Bottom left
    pts.push_back(cv::Point((source.cols * (1 - trapezoidTopWidth * smallTopWidth)) / 2, source.rows - source.rows * trapezoidHeight * smallHeight)); // Top left
    pts.push_back(cv::Point(source.cols - (source.cols * (1 - trapezoidTopWidth * smallTopWidth)) / 2, source.rows - source.rows * trapezoidHeight * smallHeight)); // Top right
    pts.push_back(cv::Point(source.cols - (source.cols * (1 - trapezoidBottomWidth + smallBottomWidth)) / 2, source.rows * bar)); // Bottom right

    // Create the mask
    cv::Mat mask = cv::Mat::zeros(source.size(), source.type());
    fillPoly(mask, pts, cv::Scalar(255, 255, 255));

    // Allocate GPU memory and copy data
    uchar* d_mask;
    uchar* d_source;
    uchar* d_result;

    cudaMalloc((void**)&d_mask, source.rows * source.cols * sizeof(uchar));
    cudaMalloc((void**)&d_source, source.rows * source.cols * sizeof(uchar));
    cudaMalloc((void**)&d_result, source.rows * source.cols * sizeof(uchar));

    cudaMemcpy(d_mask, mask.data, source.rows * source.cols * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source, source.data, source.rows * source.cols * sizeof(uchar), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((source.cols + block.x - 1) / block.x, (source.rows + block.y - 1) / block.y);

    // Call the bitwiseAndCUDA kernel
    bitwiseAndCUDA << <grid, block >> > (d_mask, d_source, d_result, source.cols, source.rows);

    cv::Mat result(source.size(), source.type());

    // Copy the result back to the host
    cudaMemcpy(result.data, d_result, source.rows * source.cols * sizeof(uchar), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_mask);
    cudaFree(d_source);
    cudaFree(d_result);

    auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds
    std::cout << "Region Masking completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return result;
}

// START: for Hough Transform Probabilistics

__global__ void houghLinesCUDA(const uchar* image, int width, int height, float rho, float theta,
    int threshold, int lineLength, int lineGap, int* accum, int numangle, int numrho, int* lines, int linesMax) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= width * height) return;

    int x = idx % width;
    int y = idx / width;

    if (image[idx] == 0) return;

    for (int n = 0; n < numangle; n++) {
        float r = x * cosf(n * theta) + y * sinf(n * theta);
        int ridx = roundf((r + numrho - 1) / rho);

        atomicAdd(&accum[n * numrho + ridx], 1);
    }

    __syncthreads();

    if (idx == 0) {
        for (int ridx = 0; ridx < numrho; ridx++) {
            for (int n = 0; n < numangle; n++) {
                int accumVal = accum[n * numrho + ridx];
                if (accumVal >= threshold) {
                    int count = 0;
                    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
                    int r = ridx - (numrho - 1) / 2;
                    float a = cosf(n * theta);
                    float b = sinf(n * theta);
                    int x2, y2;

                    x0 = static_cast<int>(x + lineLength * (-b));
                    y0 = static_cast<int>(y + lineLength * (a));
                    x1 = static_cast<int>(x - lineLength * (-b));
                    y1 = static_cast<int>(y - lineLength * (a));

                    x2 = x0;
                    y2 = y0;

                    while (true) {
                        if (x2 < 0 || x2 >= width || y2 < 0 || y2 >= height) break;

                        int idx2 = y2 * width + x2;
                        if (image[idx2] == 0) break;

                        count++;
                        x2 += static_cast<int>(-b);
                        y2 += static_cast<int>(a);
                    }

                    if (count >= lineGap) {
                        int lineIdx = atomicAdd(lines, 4);
                        if (lineIdx < linesMax) {
                            lines[lineIdx] = x0;
                            lines[lineIdx + 1] = y0;
                            lines[lineIdx + 2] = x1;
                            lines[lineIdx + 3] = y1;
                        }
                    }
                }
            }
        }
    }
}

const int linesMax = 100;

void HoughLinesProbabilistic(cv::Mat& image, float rho, float theta, int threshold,
    int lineLength, int lineGap, std::vector<cv::Vec4i>& lines) {
    // Convert the image to grayscale
    cv::Mat grayImage = image.clone();

    int width = grayImage.cols;
    int height = grayImage.rows;
    int numangle = cvFloor((CV_PI - 0.0) / theta) + 1;

    if (numangle > 1 && fabs(CV_PI - (numangle - 1) * theta) < theta / 2) {
        --numangle;
    }

    int numrho = cvRound(((width + height) * 2 + 1) / rho);

    uchar* d_image;
    int* d_accum;
    int* d_lines;

    CUDA_CHECK(cudaMalloc((void**)&d_image, width * height * sizeof(uchar)));
    CUDA_CHECK(cudaMalloc((void**)&d_accum, numangle * numrho * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_lines, linesMax * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_image, grayImage.data, width * height * sizeof(uchar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_accum, 0, numangle * numrho * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_lines, 0, linesMax * sizeof(int)));

    dim3 blockDims(256);
    dim3 gridDims((width * height + blockDims.x - 1) / blockDims.x);

    houghLinesCUDA << <gridDims, blockDims >> > (d_image, width, height, rho, theta,
        threshold, lineLength, lineGap, d_accum, numangle, numrho, d_lines, linesMax);

    CUDA_CHECK(cudaDeviceSynchronize());

    int h_lines[linesMax];
    CUDA_CHECK(cudaMemcpy(h_lines, d_lines, linesMax * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < linesMax; i += 4) {
        int x0 = h_lines[i];
        int y0 = h_lines[i + 1];
        int x1 = h_lines[i + 2];
        int y1 = h_lines[i + 3];
        if (x0 == 0 && y0 == 0 && x1 == 0 && y1 == 0) {
            break;
        }
        lines.push_back(cv::Vec4i(x0, y0, x1, y1));
    }

    cudaFree(d_image);
    cudaFree(d_accum);
    cudaFree(d_lines);
}

void customHoughLinesP(cv::Mat canny, std::vector<cv::Vec4i>& linesP,
    double rho, double theta, int thresh,
    double minLineLength, double maxGapLength)
{
    linesP.clear(); // Clear the output vector to ensure it's empty.

    std::vector<cv::Vec4i> lines;
    HoughLinesProbabilistic(canny, (float)rho, (float)theta, thresh, cvRound(minLineLength), cvRound(maxGapLength), lines);

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
    Mat gBlur = applyGaussianBlur(gray, 3);

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
    std::cout << "(CUDA: 16 threads)" << std::endl;
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