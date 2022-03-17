#include <iostream>
#include "AffineSIFT.h"
#include <string.h>
#include <vector>

using namespace std;
using namespace cv;
using algorithm::shapematch::AffineSIFT;
using algorithm::shapematch::ComputerMatch;

int main()
{
 string argv1 = "C:\\Users\\DELL\\ZED\\2\\DepthViewer_Right_22182765_720_15-03-2022-15-35-26.png";
 string argv2 = "C:\\Users\\DELL\\ZED\\1\\DepthViewer_Right_22182765_720_15-03-2022-15-34-41.png";
 string argv3 = "C:\\Users\\DELL\\ZED\\2\\result1.png";
 string argv4 = "C:\\Users\\DELL\\ZED\\2\\result2.png";
 string argv5 = "C:\\Users\\DELL\\ZED\\2\\result2.txt";
 string argv6 = "C:\\Users\\DELL\\ZED\\2\\result1-1.txt";
 string argv7 = "C:\\Users\\DELL\\ZED\\2\\result1-2.txt";
 string argv8 = "1";

 string save_path = "C:\\Users\\DELL\\ZED\\2\\result2.png";
 string savekeypts_path =  "C:\\Users\\DELL\\ZED\\2\\result1.xml";

 AffineSIFT asift_detector;
 int flag_resize = 1;
 int num_of_tilts1 = 7;
 asift_detector.setparas(save_path, flag_resize, num_of_tilts1,false,true,0.5);
 ComputerMatch src_match_keypts;
 Mat src_img = imread(argv1,0);
 //bool detect_src_flag = asift_detector.detectkeypts(src_img, savekeypts_path, src_match_keypts);

 ComputerMatch target_match_keypts;
 Mat target_img = imread(argv2,0);
 savekeypts_path = "";
 //bool detect_target_flag = asift_detector.detectkeypts(target_img, savekeypts_path, target_match_keypts);

 
 vector<string> savekeypts_path_vec; 
 savekeypts_path = "C:\\Users\\DELL\\ZED\\2\\result1.xml";
 savekeypts_path_vec.push_back(savekeypts_path);
 bool detect_flag = asift_detector.detectMatch(src_img, target_img, savekeypts_path_vec);

 
 getchar();
 return 1;

}