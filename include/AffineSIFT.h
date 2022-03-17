#pragma once
#include <vector>
#include "opencv2/opencv.hpp"
#include "demo_lib_sift.h"
namespace algorithm {
namespace shapematch {
 using namespace std;
 using namespace cv;
 class  ComputerMatch {
 public:
  ComputerMatch();
  ~ComputerMatch();
  float zoom;
  int num_of_tilts;
  int wSize;
  int hSize;
  int verb;
  int num_keys;
  vector<vector<keypointslist>> keyspts;
  siftPar siftparameters;
 };

class AffineSIFT {
 public:
  AffineSIFT();
  ~AffineSIFT();

  bool setparas(const string & save_path, const int & flag_resize, const int & num_of_tilts1,
   const bool&read_keypts_flag, const bool &show_matchresult,const float& scale_factor);

  bool detectkeypts(const Mat & src_img, const string & savekeypts_path, ComputerMatch & match_keypts, siftPar& sift_parameters);

  bool detectMatch(const Mat & src_img, const Mat & target_img, const vector<string>& savekeypts_path);

  string save_keypts_;
  string save_path_;//������ͼ��·��
  int flag_resize_;//�Ƿ�����
  int num_of_tilts_;//�ӽǸ���
  bool read_keypts_flag_;//�Ƿ���ؼ����ļ�
  bool show_matchresult_;//�Ƿ���ʾƥ����ͼ��
  float scale_factor_;//ͼ�����ű�
  ComputerMatch src_keypts_;
  ComputerMatch target_keypts_;
  vector<Point2f> src_pts_;
  vector<Point2f> target_pts_;
};
}
}
