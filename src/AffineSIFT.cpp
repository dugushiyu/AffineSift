#include "AffineSIFT.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <core/core.hpp>
#include "library.h"
#include "frot.h"
#include "fproj.h"
#include "compute_asift_keypoints.h"
#include "compute_asift_matches.h"

#ifdef _OPENMP
#include <omp.h>
#include "AffineSIFT.h"
#endif

namespace algorithm {
namespace shapematch {
using namespace std;
using namespace cv;
# define IM_X 800
# define IM_Y 600
AffineSIFT::AffineSIFT() {
  read_keypts_flag_ = false;
  show_matchresult_ = false;
}

AffineSIFT::~AffineSIFT() {
}

template<typename _Tp>
vector<_Tp> convertMat2Vector(const Mat &mat) {
  return (vector<_Tp>)(mat.reshape(1, 1));//通道数不变，按行转为一行
}

bool AffineSIFT::setparas(const string& save_path, const int&flag_resize,
                          const int&num_of_tilts1, const bool&read_keypts_flag,
 const bool &show_matchresult,const float& scale_factor) {
  save_path_ = save_path;
  flag_resize_ = flag_resize;
  num_of_tilts_ = num_of_tilts1;
  read_keypts_flag_ = read_keypts_flag;
  show_matchresult_ = show_matchresult;
  scale_factor_ = scale_factor;
  return true;
}
bool AffineSIFT::detectkeypts(const Mat& src_img, const string& savekeypts_path, ComputerMatch &match_keypts, siftPar& sift_parameters) {
  if (src_img.empty()) {
    return false;
  }
  // 1.格式转换
  Mat img1_in_f;
  src_img.convertTo(img1_in_f, CV_32F);
  size_t w1 = img1_in_f.cols, h1 = img1_in_f.rows;
  vector<float> ipixels1 = convertMat2Vector<float>(img1_in_f);
  //2. Resize the images to area wS*hW in remaining the apsect-ratio
  // Resize if the resize flag is not set or if the flag is set unequal to 0
  float wS = src_img.cols * scale_factor_;// IM_X;
  float hS = src_img.rows * scale_factor_;//IM_Y;
  float zoom1 = 0;
  int wS1 = 0, hS1 = 0;
  vector<float> ipixels1_zoom;
  int flag_resize = flag_resize_;
  if (flag_resize != 0) { //缩放到固定大小但是结果是原图上
    cout << "WARNING: The input images are resized to " << wS << "x" << hS << " for ASIFT. " << endl
         << "         But the results will be normalized to the original image size." << endl;
    float InitSigma_aa = 1.6;
    float fproj_p, fproj_bg;
    char fproj_i;
    float *fproj_x4, *fproj_y4;
    int fproj_o;
    fproj_o = 3;
    fproj_p = 0;
    fproj_i = 0;
    fproj_bg = 0;
    fproj_x4 = 0;
    fproj_y4 = 0;
    float areaS = wS * hS;
// Resize image 1
    float area1 = w1 * h1;
    zoom1 = sqrt(area1 / areaS);
    wS1 = (int)(w1 / zoom1);
    hS1 = (int)(h1 / zoom1);
    int fproj_sx = wS1;
    int fproj_sy = hS1;
    float fproj_x1 = 0;
    float fproj_y1 = 0;
    float fproj_x2 = wS1;
    float fproj_y2 = 0;
    float fproj_x3 = 0;
    float fproj_y3 = hS1;
    /* Anti-aliasing filtering along vertical direction */
    if (zoom1 > 1) {
      float sigma_aa = InitSigma_aa * zoom1 / 2;
      GaussianBlur1D(ipixels1, w1, h1, sigma_aa, 1);
      GaussianBlur1D(ipixels1, w1, h1, sigma_aa, 0);
    }
   // simulate a tilt: subsample the image along the vertical axis by a factor of t.
    ipixels1_zoom.resize(wS1 * hS1);
    fproj(ipixels1, ipixels1_zoom, w1, h1, &fproj_sx, &fproj_sy, &fproj_bg, &fproj_o, &fproj_p,
          &fproj_i, fproj_x1, fproj_y1, fproj_x2, fproj_y2, fproj_x3, fproj_y3, fproj_x4, fproj_y4);
  } else { //不缩放
    ipixels1_zoom.resize(w1 * h1);
    ipixels1_zoom = ipixels1;
    wS1 = w1;
    hS1 = h1;
    zoom1 = 1;
  }
// 3.计算特征关键点
  int num_of_tilts1 = num_of_tilts_;
  int verb = 0; // show / don not show verbose messages. (1 for debugging)
  vector< vector< keypointslist > > keys1;
  cout << "Computing keypoints on the two images..." << endl;
  time_t tstart, tend;
  tstart = time(0);
  int num_keys1 = compute_asift_keypoints(ipixels1_zoom, wS1, hS1, num_of_tilts1, verb, keys1, sift_parameters);
  tend = time(0);
  cout << "Keypoints computation accomplished in " << difftime(tend, tstart) << " seconds." << endl;
  // 4.存储结构体
  match_keypts.num_of_tilts = num_of_tilts1;
  match_keypts.wSize = wS1;
  match_keypts.hSize = hS1;
  match_keypts.verb = verb;
  match_keypts.zoom = zoom1;
  match_keypts.num_keys = num_keys1;
  match_keypts.siftparameters = sift_parameters;
  //match_keypts.keyspts = keys1;
  match_keypts.keyspts.assign(keys1.begin(), keys1.end());
  if (savekeypts_path.length() > 1) {
    cv::FileStorage fs(savekeypts_path, cv::FileStorage::WRITE);
    bool open_flag = fs.isOpened();
    int keys_size = (int)keys1.size();
    fs << "num_of_tilts" << num_of_tilts1;
    fs << "wSize" << wS1;
    fs << "hSize" << hS1;
    fs << "verb" << verb;
    fs << "zoom" << zoom1;
    fs << "num_keys" << num_keys1;
    fs << "keys1_size" << keys_size;
    // Follow the same convention of David Lowe:
    // the first line contains the number of keypoints and the length of the desciptors (128)
    for (int tt = 0; tt < keys_size; tt++) {
      string node_first = to_string(tt);
      node_first = "tilts_first_" + node_first;
      fs << node_first << "{";
      for (int rr = 0; rr < (int)keys1[tt].size(); rr++) {
       string node_second = to_string(rr);
       node_second = "tilts_second_" + node_second;
       fs << node_second << "{";
        keypointslist::iterator ptr = keys1[tt][rr].begin();
        for (int i = 0; i < (int)keys1[tt][rr].size(); i++, ptr++) {
          string vec_num = to_string(i);
          fs << "x" + vec_num << zoom1*ptr->x << "y" + vec_num << zoom1*ptr->y << "scale" + vec_num << zoom1*ptr->scale << "angle" + vec_num << ptr->angle << "vec" + vec_num << "[";
          for (int ii = 0; ii < (int)VecLength; ii++) {
            int ptr_vec_val = (int)ptr->vec[ii];
            fs << ptr_vec_val;
          }
          fs << "]";
        }
        fs << "}";
      }
      fs << "}";
    }
    fs.release();
  }
  return true;
}

bool AffineSIFT::detectMatch(const Mat & src_img, const Mat & target_img, const vector<string>& savekeypts_path) {
  if ((savekeypts_path.size() < 1 && src_img .empty()) || target_img.empty()) {
    return false;
  }
  // 1.计算特征点
  string  target_keypt_path;
  ComputerMatch src_match_keypts, target_match_keypts;
  string src_keypt_path = savekeypts_path[0];
  // Define the SIFT parameters
  siftPar siftparameters;
  default_sift_parameters(siftparameters);
  {
    if (read_keypts_flag_ = 0) {
      int keyptsv1 = 0, keyptsv2 = 0;
      FileStorage fs_read(src_keypt_path, FileStorage::READ);
      fs_read["num_of_tilts"] >> src_match_keypts.num_of_tilts;
      fs_read["wSize"] >> src_match_keypts.wSize;
      fs_read["hSize"] >> src_match_keypts.hSize;
      fs_read["verb"] >> src_match_keypts.verb;
      fs_read["zoom"] >> src_match_keypts.zoom;
      fs_read["num_keys"] >> src_match_keypts.num_keys;
      fs_read["keys1_size"] >> keyptsv1;
      int telts_num = src_match_keypts.num_of_tilts;
      int num_tile = 0;
     
      for (int i = 0; i < keyptsv1; i++) {//0-6
        string node_first = to_string(i);
        node_first = "tilts_first_" + node_first;
        FileNode tm_first = fs_read[node_first];
        FileNodeIterator it_first = tm_first.begin();
        vector<vector<keypoint>> tem_keypts_vec;
        for (int k=0; it_first!= tm_first.end(); it_first++,k++) { 
         string node_second = to_string(k);
         node_second = "tilts_first_" + node_second;
         FileNode tm_second = fs_read[node_second];
         FileNodeIterator it_second = tm_second.begin();
        vector<keypoint> tem_keypts_list;
        for (int t = 0; t < 10000, it_second != tm_second.end(); t++, it_second++) {
          keypoint tem_keypt;
          string node_second = to_string(t);
          tem_keypt.x = (float)(*it_second)["x" + node_second];
          tem_keypt.y = (float)(*it_second)["y" + node_second];
          tem_keypt.scale = (float)(*it_second)["scale" + node_second];
          tem_keypt.angle = (float)(*it_second)["angle" + node_second];
          FileNodeIterator it_vec = (*it_second)["vec" + node_second].begin();
          for (int j = 0; j < VecLength, it_vec != (*it_second)["vec" + node_second].end(); j++, it_vec++) {
            tem_keypt.vec[j] = (int)*it_vec;
          }
          tem_keypts_list.push_back(tem_keypt);
        }       
        tem_keypts_vec.push_back(tem_keypts_list);
        } 
        src_match_keypts.keyspts.push_back(tem_keypts_vec);
      }
      src_match_keypts.siftparameters = siftparameters;//加载文件少了参数优化
      fs_read.release();
    } else {
      detectkeypts(src_img, src_keypt_path, src_match_keypts, siftparameters);
    }
    detectkeypts(target_img, target_keypt_path, target_match_keypts, siftparameters);
  }
// 2.Match ASIFT keypoints
  matchingslist matchings;
  cout << "Matching the keypoints..." << endl;
  time_t tstart = time(0);
  int num_matchings = compute_asift_matches(src_match_keypts.num_of_tilts, target_match_keypts.num_of_tilts
                      , src_match_keypts.wSize, src_match_keypts.hSize, target_match_keypts.wSize, target_match_keypts.hSize,
                      src_match_keypts.verb, src_match_keypts.keyspts, target_match_keypts.keyspts,
                      matchings, siftparameters);
  time_t tend = time(0);
  cout << "Keypoints matching accomplished in " << difftime(tend, tstart) << " seconds." << endl;
// 3.Output image containing line matches (the two images are concatenated one above the other)
  if (show_matchresult_) {
    int match_nums = (int)matchings.size();
    Mat resultshowv, resultshowh;
    vconcat(src_img, target_img, resultshowv);
    hconcat(src_img, target_img, resultshowh);
    matchingslist::iterator ptr = matchings.begin();
    Mat resultmatchpoints = Mat::zeros(Size(4, match_nums), CV_32F);  
    float zoom1 = src_match_keypts.zoom;
    float zoom2 = target_match_keypts.zoom;
    int h1 = src_img.rows;
    int w1 = src_img.cols;
    for (int i = 0; i < match_nums; i++, ptr++) {
      Point v1_s1{ (int)(zoom1 * ptr->first.x), (int)(zoom1 * ptr->first.y) };
      Point v2_e1{ (int)(zoom2 * ptr->second.x), (int)(zoom2 * ptr->second.y + h1) };
      line(resultshowv, v1_s1, v2_e1, Scalar(0, 0, 255), 2);
      Point h1_s1{ (int)(zoom1 * ptr->first.x), (int)(zoom1 * ptr->first.y) };
      Point h2_e1{ (int)(zoom2 * ptr->second.x + w1), (int)(zoom2 * ptr->second.y) };
      line(resultshowh, h1_s1, h2_e1, Scalar(0, 0, 255), 2);
      resultmatchpoints.at<float>(i, 0) = v1_s1.x;
      resultmatchpoints.at<float>(i, 1) = v1_s1.y;
      resultmatchpoints.at<float>(i, 2) = v2_e1.x;
      resultmatchpoints.at<float>(i, 3) = h2_e1.y;
      src_pts_.push_back({ (zoom1 * ptr->first.x), (zoom1 * ptr->first.y) });
      target_pts_.push_back({ (zoom2 * ptr->second.x), (zoom2 * ptr->second.y) });
    }
    //////////////////////////////////////////////////////////////////////////
    // 单映性矩阵
    Mat monomorphic_matrix = findHomography(src_pts_, target_pts_);
    //3.原图到目标图片的变换
    Mat im_out_perspective;
    warpPerspective(src_img, im_out_perspective, monomorphic_matrix, target_img.size());// 透视变换
                                                                              //
    Mat  im_out_affine;
    Mat rigid_matrix = estimateRigidTransform(src_pts_, target_pts_, true);
    warpAffine(src_img, im_out_affine, rigid_matrix, target_img.size());
    //////////////////////////////////////////////////////////////////////////

    if (save_path_.length() > 1) {
      imwrite(save_path_, resultshowh);
    }
  }
  return true;
}


ComputerMatch::ComputerMatch() {
  keyspts.clear();
}


ComputerMatch::~ComputerMatch() {
}
}
}

