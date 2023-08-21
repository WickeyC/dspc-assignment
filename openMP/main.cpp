#include <iostream>
#include <iomanip> 
#include <opencv2/opencv.hpp>
#include <chrono> // For measuring time

//For suppressing info logs in debug mode
#include <opencv2/core/utils/logger.hpp>

using namespace cv;

Mat filterColors(Mat source, bool isDayTime)
{
    Mat hsv, whiteMask, whiteImage, yellowMask, yellowImage, whiteYellow;

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
        return dst;
    }

    // Return white and yellow mask if image is taken during the day
    return whiteYellow;
}

// Function to convert an image to grayscale
Mat convertToGrayscale(const Mat& source) {
    Mat grayImage(source.size(), CV_8UC1);

    auto startTime = std::chrono::high_resolution_clock::now(); // Record the start time

    for (int row = 0; row < source.rows; ++row) {
        for (int col = 0; col < source.cols; ++col) {
            Vec3b pixel = source.at<Vec3b>(row, col);
            uchar grayValue = static_cast<uchar>((pixel[0] + pixel[1] + pixel[2]) / 3);
            grayImage.at<uchar>(row, col) = grayValue;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds

    std::cout << "Grayscale conversion completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return grayImage;
}

// Function for Gaussian Smoothing
Mat applyGaussianSmoothing(const Mat& source, int kernelSize, double sigma) {
    Mat smoothedImage = source.clone();

    int radius = kernelSize / 2;
    int size = kernelSize * kernelSize;

    // Generate Gaussian kernel
    std::vector<std::vector<double>> kernel(kernelSize, std::vector<double>(kernelSize));
    double kernelSum = 0.0;

    for (int x = -radius; x <= radius; ++x) {
        for (int y = -radius; y <= radius; ++y) {
            kernel[x + radius][y + radius] = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * CV_PI * sigma * sigma);
            kernelSum += kernel[x + radius][y + radius];
        }
    }

    // Normalize the kernel
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel[i][j] /= kernelSum;
        }
    }

    // Apply convolution with the kernel
    for (int i = radius; i < source.rows - radius; ++i) {
        for (int j = radius; j < source.cols - radius; ++j) {
            double newValue = 0.0;

            for (int x = -radius; x <= radius; ++x) {
                for (int y = -radius; y <= radius; ++y) {
                    newValue += source.at<uchar>(i + x, j + y) * kernel[x + radius][y + radius];
                }
            }

            smoothedImage.at<uchar>(i, j) = static_cast<uchar>(newValue);
        }
    }

    return smoothedImage;
}

// Function for Canny Edge Detection
Mat applyCannyEdgeDetection(const Mat& source, double lowThreshold, double highThreshold) {
    Mat edges = source.clone();

    auto startTime = std::chrono::high_resolution_clock::now(); // Record the start time

    // Apply Gaussian smoothing before edge detection
    GaussianBlur(source, edges, Size(5, 5), 1.4);

    // Calculate gradients using Sobel filters
    Mat gradientX, gradientY;
    Sobel(edges, gradientX, CV_64F, 1, 0, 3);
    Sobel(edges, gradientY, CV_64F, 0, 1, 3);

    // Calculate gradient magnitude and direction
    Mat gradientMagnitude, gradientDirection;
    magnitude(gradientX, gradientY, gradientMagnitude);
    phase(gradientX, gradientY, gradientDirection, true);

    // Non-maximum suppression
    Mat suppressedEdges = Mat::zeros(gradientMagnitude.size(), CV_8U);

    for (int i = 1; i < gradientMagnitude.rows - 1; ++i) {
        for (int j = 1; j < gradientMagnitude.cols - 1; ++j) {
            double angle = gradientDirection.at<double>(i, j);
            angle = angle < 0 ? angle + CV_PI : angle;
            angle = angle * 180.0 / CV_PI;

            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                if (gradientMagnitude.at<double>(i, j) > gradientMagnitude.at<double>(i, j - 1) &&
                    gradientMagnitude.at<double>(i, j) > gradientMagnitude.at<double>(i, j + 1)) {
                    suppressedEdges.at<uchar>(i, j) = static_cast<uchar>(gradientMagnitude.at<double>(i, j));
                }
            }
            // ... Repeat for other angles
        }
    }

    // Hysteresis thresholding
    Mat edgesFinal = Mat::zeros(suppressedEdges.size(), CV_8U);

    for (int i = 1; i < suppressedEdges.rows - 1; ++i) {
        for (int j = 1; j < suppressedEdges.cols - 1; ++j) {
            if (suppressedEdges.at<uchar>(i, j) >= highThreshold) {
                edgesFinal.at<uchar>(i, j) = 255;
            }
            else if (suppressedEdges.at<uchar>(i, j) >= lowThreshold) {
                // Check 8 neighboring pixels
                if (suppressedEdges.at<uchar>(i - 1, j - 1) == 255 ||
                    suppressedEdges.at<uchar>(i - 1, j) == 255 ||
                    suppressedEdges.at<uchar>(i - 1, j + 1) == 255 ||
                    suppressedEdges.at<uchar>(i, j - 1) == 255 ||
                    suppressedEdges.at<uchar>(i, j + 1) == 255 ||
                    suppressedEdges.at<uchar>(i + 1, j - 1) == 255 ||
                    suppressedEdges.at<uchar>(i + 1, j) == 255 ||
                    suppressedEdges.at<uchar>(i + 1, j + 1) == 255) {
                    edgesFinal.at<uchar>(i, j) = 255;
                }
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds

    std::cout << "Canny Edge Detection completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return edgesFinal;
}

// Function for Region Masking
Mat applyRegionMask(const Mat& source, const std::vector<Point>& vertices) {
    Mat maskedImage = Mat::zeros(source.size(), source.type());

    auto startTime = std::chrono::high_resolution_clock::now(); // Record the start time

    // Create a mask polygon using the provided vertices
    std::vector<std::vector<Point>> maskPolygons = { vertices };
    fillPoly(maskedImage, maskPolygons, Scalar(255, 255, 255));

    // Apply the mask to the source image
    Mat maskedResult;
    bitwise_and(source, maskedImage, maskedResult);

    auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds

    std::cout << "Region Masking completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return maskedResult;
}

// Function for Hough Transform
std::vector<std::pair<double, double>> applyHoughTransform(const Mat& edges, int rhoSteps, int thetaSteps, int threshold) {
    std::vector<std::pair<double, double>> lines;

    auto startTime = std::chrono::high_resolution_clock::now(); // Record the start time

    int width = edges.cols;
    int height = edges.rows;
    double maxRho = sqrt(width * width + height * height);
    double deltaRho = maxRho / rhoSteps;
    double deltaTheta = CV_PI / thetaSteps;

    std::vector<int> accumulator(rhoSteps * thetaSteps, 0);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (edges.at<uchar>(y, x) > 0) {
                for (int tIndex = 0; tIndex < thetaSteps; ++tIndex) {
                    double theta = tIndex * deltaTheta;
                    double rho = x * cos(theta) + y * sin(theta);
                    int rIndex = static_cast<int>((rho + maxRho) / deltaRho);
                    accumulator[rIndex * thetaSteps + tIndex]++;
                }
            }
        }
    }

    for (int rIndex = 0; rIndex < rhoSteps; ++rIndex) {
        for (int tIndex = 0; tIndex < thetaSteps; ++tIndex) {
            if (accumulator[rIndex * thetaSteps + tIndex] > threshold) {
                double rho = (rIndex * deltaRho) - maxRho;
                double theta = tIndex * deltaTheta;
                lines.push_back(std::make_pair(rho, theta));
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds

    std::cout << "Hough Transform completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return lines;
}

// Function to draw lines on an image
void drawLines(Mat& image, const std::vector<std::pair<double, double>>& lines) {
    for (const auto& line : lines) {
        double rho = line.first;
        double theta = line.second;

        double a = cos(theta);
        double b = sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;

        Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));

        cv::line(image, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
    }
}

int main() {
    // Control OpenCV Log in Debug Mode (For a clearer console)
    // Suppress all Log
     //utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
    // Only shows Error level log
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

    // Load the image
    Mat originalImage = imread("../images/road1.png");


    if (originalImage.empty()) {
        std::cerr << "Error: Could not read image'" << std::endl;
        return -1;
    }

    // Convert the image to grayscale using your implementation
    Mat grayscaleImage = convertToGrayscale(originalImage);

    // Apply Gaussian Smoothing
    int kernelSize = 9; // Adjust as needed
    double sigma = 1.0; // Adjust as needed
    Mat smoothedImage = applyGaussianSmoothing(grayscaleImage, kernelSize, sigma);

    // Apply Canny Edge Detection
    double lowThreshold = 30.0; // Adjust as needed
    double highThreshold = 100.0; // Adjust as needed
    Mat edges = applyCannyEdgeDetection(smoothedImage, lowThreshold, highThreshold);

    // Apply Region Masking
    std::vector<Point> regionVertices = {
        Point(0, edges.rows),
        Point(edges.cols / 2, edges.rows / 2),
        Point(edges.cols, edges.rows)
    };
    Mat maskedEdges = applyRegionMask(edges, regionVertices);

    // Apply Hough Transform
    int rhoSteps = 1; // Adjust as needed
    int thetaSteps = 180; // Adjust as needed
    int threshold = 50; // Adjust as needed
    std::vector<std::pair<double, double>> lines = applyHoughTransform(maskedEdges, rhoSteps, thetaSteps, threshold);

    // Draw detected lines on the original image
    Mat imageWithLines = originalImage.clone();
    drawLines(imageWithLines, lines);

    // Display the images
    namedWindow("Original Image", WINDOW_NORMAL);
    namedWindow("Grayscale Image", WINDOW_NORMAL);
    namedWindow("Smoothed Image", WINDOW_NORMAL);
    namedWindow("Canny Edges", WINDOW_NORMAL);
    namedWindow("Masked Edges", WINDOW_NORMAL);
    namedWindow("Image with Lines", WINDOW_NORMAL);

    imshow("Original Image", originalImage);
    imshow("Grayscale Image", grayscaleImage);
    imshow("Smoothed Image", smoothedImage);
    imshow("Canny Edges", edges);
    imshow("Masked Edges", maskedEdges);
    imshow("Image with Lines", imageWithLines);

    waitKey(0);

    return 0;
}
int main(int argc, char* argv[])
{
    // Control OpenCV Log in Debug Mode (For a clearer console)
    // Suppress all Log
    //utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
    // Only shows Error level log
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

    // Load source video
    if (argc != 2) {
        std::cout << "Usage: ./exe path-to-video" << std::endl;
        return -1;
    }

    // Initialize video capture for reading a videofile
    VideoCapture cap(argv[1]);

    // Check if video can be opened
    if (!cap.isOpened())
    {
        std::cout << "Failed to open videofile!" << std::endl;
        return -1;
    }

    // Read and analyze video
    while (true)
    {
        Mat frame;
        cap >> frame;

        // Stop if frame is empty (end of video)
        if (frame.empty())
        {
            break;
        }

        // Determine if video is taken during daytime or not
        bool isDay = isDayTime(frame);

        // Filter image 
        Mat filteredIMG = filterColors(frame, isDay);

        // Apply grayscale
        Mat gray = applyGrayscale(filteredIMG);

        // Apply Gaussian blur
        Mat gBlur = applyGaussianBlur(gray);

        // Find edges
        Mat edges = applyCanny(gBlur);

        // Create mask (Region of Interest)
        Mat maskedIMG = RegionOfInterest(edges);

        // Detect straight lines and draw the lanes if possible
        std::vector<Vec4i> linesP = houghLines(maskedIMG, frame.clone(), false);
        Mat lanes = drawLanes(frame, linesP);
        imshow("Lanes", lanes);

        // Press  ESC on keyboard to exit
        if (waitKey(5) == 27) break;
    }

    cap.release();

    return 0;
}