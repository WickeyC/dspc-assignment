#include <iostream>
#include <iomanip> 
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <chrono> // For measuring time

//For suppressing info logs in debug mode
#include <opencv2/core/utils/logger.hpp>

using namespace cv;

#define NTHREAD 6

// Function to convert an image to grayscale (using your own implementation)
Mat convertToGrayscaleOpenMP(const Mat& source) {
    Mat grayImage(source.size(), CV_8UC1);

    double startTime = omp_get_wtime(); // Record the start time

    #pragma omp parallel num_threads(NTHREAD)
    {
        int numRows = source.rows;

        #pragma omp for schedule(dynamic)
        for (int row = 0; row < numRows; ++row) {
            Vec3b pixel;
            uchar grayValue;

            for (int col = 0; col < source.cols; ++col) {
                pixel = source.at<Vec3b>(row, col);
                grayValue = static_cast<uchar>((pixel[0] + pixel[1] + pixel[2]) / 3);
                grayImage.at<uchar>(row, col) = grayValue;
            }
        }
    }

    double endTime = omp_get_wtime(); // Record the end time
    double executionTime = endTime - startTime;

    std::cout << "OpenMP conversion completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return grayImage;
}

// Function to convert an image to grayscale without OpenMP
Mat convertToGrayscaleSerial(const Mat& source) {
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

    std::cout << "Serial conversion completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return grayImage;
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
    Mat grayscaleImageOpenMP = convertToGrayscaleOpenMP(originalImage);
    Mat grayscaleImageSerial = convertToGrayscaleSerial(originalImage);

    // Display the original and grayscale images
    namedWindow("Original Image", WINDOW_NORMAL);
    namedWindow("OpenMP Grayscale Image", WINDOW_NORMAL);
    namedWindow("Serial Grayscale Image", WINDOW_NORMAL);

    imshow("Original Image", originalImage);
    imshow("OpenMP Grayscale Image", grayscaleImageOpenMP);
    imshow("Serial Grayscale Image", grayscaleImageSerial);

    // Save the grayscale image
    imwrite("grayscale_road_OpenMP.png", grayscaleImageOpenMP);
    imwrite("grayscale_road_Serial.png", grayscaleImageSerial);

    waitKey(0);

    return 0;
}
