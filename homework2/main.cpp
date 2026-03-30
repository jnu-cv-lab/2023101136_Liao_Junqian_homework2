#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

string type2str(int type) {
    string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');
    return r;
}

int main() {
    //读取测试图片
    string img_path = "test.jpg";
    Mat img = imread(img_path, IMREAD_COLOR);
    if (img.empty()) {
        cout << " 无法读取图片，请检查路径是否正确" << endl;
        return -1;

    }

    //输出图像基本信息
    cout << "===== 图像基本信息 =====" << endl;
    cout << "宽度 (Width): " << img.cols << " 像素" << endl;
    cout << "高度 (Height): " << img.rows << " 像素" << endl;
    cout << "通道数 (Channels): " << img.channels() << endl;
    string type_str = type2str(img.type());
    cout << "数据类型 (Data Type): " << type_str << endl;
    cout << "图像尺寸 (Size): " << img.size() << endl;
    cout << "========================" << endl << endl;

    //显示原图
    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", img);
    waitKey(0);
    destroyWindow("Original Image");

    //转换为灰度图并显示
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    namedWindow("Grayscale Image", WINDOW_NORMAL);
    imshow("Grayscale Image", gray_img);
    waitKey(0);
    destroyWindow("Grayscale Image");

    //保存灰度图
    string gray_save_path = "gray_test.jpg";
    bool is_saved = imwrite(gray_save_path, gray_img);
    if (is_saved) {
        cout << "灰度图已成功保存至: " << gray_save_path << endl;
    } else {
        cout << "灰度图保存失败" << endl;
    }
    cout << endl;

    cout << "===== 像素值操作 =====" << endl;
    Vec3b color_pixel = img.at<Vec3b>(100, 100); 
    cout << "彩色图(100, 100)像素值 (B, G, R): " 
         << (int)color_pixel[0] << ", " << (int)color_pixel[1] << ", " << (int)color_pixel[2] << endl;
    uchar gray_pixel = gray_img.at<uchar>(100, 100); 
    cout << "灰度图(100, 100)像素值: " << (int)gray_pixel << endl;
    cout << "=====================" << endl << endl;

 
    Rect roi_rect(0, 0, 100, 100); 
    Mat roi_img = img(roi_rect);
    string roi_save_path = "cropped.jpg";
    imwrite(roi_save_path, roi_img);
    cout << "左上角100×100区域已保存至: " << roi_save_path << endl;

  
    namedWindow("Top-left ROI (100×100)", WINDOW_NORMAL);
    imshow("Top-left ROI (100×100)", roi_img);
    waitKey(0);
    destroyAllWindows();

    return 0;
}


