// Copyright (c) 2008-2011, Guoshen Yu <yu@cmap.polytechnique.fr>
// Copyright (c) 2008-2011, Jean-Michel Morel <morel@cmla.ens-cachan.fr>
//
// WARNING: 
// This file implements an algorithm possibly linked to the patent
//
// Jean-Michel Morel and Guoshen Yu, Method and device for the invariant 
// affine recognition recognition of shapes (WO/2009/150361), patent pending. 
//
// This file is made available for the exclusive aim of serving as
// scientific tool to verify of the soundness and
// completeness of the algorithm description. Compilation,
// execution and redistribution of this file may violate exclusive
// patents rights in certain countries.
// The situation being different for every country and changing
// over time, it is your responsibility to determine which patent
// rights restrictions apply to you before you compile, use,
// modify, or redistribute this file. A patent lawyer is qualified
// to make this determination.
// If and only if they don't conflict with any patent terms, you
// can benefit from the following license terms attached to this
// file.
//
// This program is provided for scientific and educational only:
// you can use and/or modify it for these purposes, but you are
// not allowed to redistribute this work or derivative works in
// source or executable form. A license must be obtained from the
// patent right holders for any other use.
//
// 
//*----------------------------- demo_ASIFT  --------------------------------*/
// Detect corresponding points in two images with the ASIFT method. 

// Please report bugs and/or send comments to Guoshen Yu yu@cmap.polytechnique.fr
// 
// Reference: J.M. Morel and G.Yu, ASIFT: A New Framework for Fully Affine Invariant Image 
//            Comparison, SIAM Journal on Imaging Sciences, vol. 2, issue 2, pp. 438-469, 2009. 
// Reference: ASIFT online demo (You can try ASIFT with your own images online.) 
//			  http://www.ipol.im/pub/algo/my_affine_sift/
/*---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
using namespace std;

#ifdef _OPENMP
#include <omp.h>
#endif

#include "demo_lib_sift.h"
//#include "io_png/io_png.h"
#include "opencv2/opencv.hpp"
#include <core/core.hpp>
#include "library.h"
#include "frot.h"
#include "fproj.h"
#include "compute_asift_keypoints.h"
#include "compute_asift_matches.h"
using namespace cv;
# define IM_X 800
# define IM_Y 600

template<typename _Tp>
vector<_Tp> convertMat2Vector(const Mat &mat)
{
 return (vector<_Tp>)(mat.reshape(1, 1));//通道数不变，按行转为一行
}

int main22(int argc, char **argv)
{			
	
    argv[0] = "";
    argv[1] = "C:\\Users\\DELL\\ZED\\2\\DepthViewer_Right_22182765_720_15-03-2022-15-35-26.png";
    argv[2] = "C:\\Users\\DELL\\ZED\\1\\DepthViewer_Right_22182765_720_15-03-2022-15-34-41.png";
    argv[3] = "C:\\Users\\DELL\\ZED\\2\\result1.png";
    argv[4] = "C:\\Users\\DELL\\ZED\\2\\result2.png";
    argv[5] = "C:\\Users\\DELL\\ZED\\2\\result2.txt";
    argv[6] = "C:\\Users\\DELL\\ZED\\2\\result1-1.txt";
    argv[7] = "C:\\Users\\DELL\\ZED\\2\\result1-2.txt";
    argv[8] = "1";
    //////////////////////////////////////////////////////////////////////////Input
    Mat img1 = imread(argv[1],0);
    Mat img1_in_f;
    img1.convertTo(img1_in_f,CV_32F);
    size_t w1= img1_in_f.cols, h1= img1_in_f.rows;
    vector<float> ipixels1 = convertMat2Vector<float>(img1_in_f);

    Mat img2 = imread(argv[2],0);
    Mat img2_in_f;
    img2.convertTo(img2_in_f, CV_32F);
    size_t w2 = img2_in_f.cols, h2 = img2_in_f.rows;
    vector<float> ipixels2 = convertMat2Vector<float>(img2_in_f);

	///// Resize the images to area wS*hW in remaining the apsect-ratio	
	///// Resize if the resize flag is not set or if the flag is set unequal to 0
	float wS = IM_X;
	float hS = IM_Y;
	
	float zoom1=0, zoom2=0;	
	int wS1=0, hS1=0, wS2=0, hS2=0;
	vector<float> ipixels1_zoom, ipixels2_zoom;	
		
	int flag_resize = 1;
	if (argc == 9)
	{	
		flag_resize = atoi(argv[8]);
	}
	
	if ((argc == 8) || (flag_resize != 0))
	{
		cout << "WARNING: The input images are resized to " << wS << "x" << hS << " for ASIFT. " << endl 
		<< "         But the results will be normalized to the original image size." << endl << endl;
		
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
		zoom1 = sqrt(area1/areaS);
		
		wS1 = (int) (w1 / zoom1);
		hS1 = (int) (h1 / zoom1);
		
		int fproj_sx = wS1;
		int fproj_sy = hS1;     
		
		float fproj_x1 = 0;
		float fproj_y1 = 0;
		float fproj_x2 = wS1;
		float fproj_y2 = 0;
		float fproj_x3 = 0;	     
		float fproj_y3 = hS1;
		
		/* Anti-aliasing filtering along vertical direction */
		if ( zoom1 > 1 )		{
			float sigma_aa = InitSigma_aa * zoom1 / 2;
			GaussianBlur1D(ipixels1,w1,h1,sigma_aa,1);
			GaussianBlur1D(ipixels1,w1,h1,sigma_aa,0);
		}
			
		// simulate a tilt: subsample the image along the vertical axis by a factor of t.
		ipixels1_zoom.resize(wS1*hS1);
		fproj (ipixels1, ipixels1_zoom, w1, h1, &fproj_sx, &fproj_sy, &fproj_bg, &fproj_o, &fproj_p, 
			   &fproj_i , fproj_x1 , fproj_y1 , fproj_x2 , fproj_y2 , fproj_x3 , fproj_y3, fproj_x4, fproj_y4); 
		
		
		// Resize image 2 
		float area2 = w2 * h2;
		zoom2 = sqrt(area2/areaS);
				
		wS2 = (int) (w2 / zoom2);
		hS2 = (int) (h2 / zoom2);
		
		fproj_sx = wS2;
		fproj_sy = hS2;     
		
		fproj_x2 = wS2;
		fproj_y3 = hS2;
		
		/* Anti-aliasing filtering along vertical direction */
		if ( zoom1 > 1 )		{
			float sigma_aa = InitSigma_aa * zoom2 / 2;
			GaussianBlur1D(ipixels2,w2,h2,sigma_aa,1);
			GaussianBlur1D(ipixels2,w2,h2,sigma_aa,0);
		}
			
		// simulate a tilt: subsample the image along the vertical axis by a factor of t.
		ipixels2_zoom.resize(wS2*hS2);
		fproj (ipixels2, ipixels2_zoom, w2, h2, &fproj_sx, &fproj_sy, &fproj_bg, &fproj_o, &fproj_p, 
			   &fproj_i , fproj_x1 , fproj_y1 , fproj_x2 , fproj_y2 , fproj_x3 , fproj_y3, fproj_x4, fproj_y4); 
	}	else 	{
		ipixels1_zoom.resize(w1*h1);	
		ipixels1_zoom = ipixels1;
		wS1 = w1;
		hS1 = h1;
		zoom1 = 1;
		
		ipixels2_zoom.resize(w2*h2);	
		ipixels2_zoom = ipixels2;
		wS2 = w2;
		hS2 = h2;
		zoom2 = 1;
	}

	
	///// Compute ASIFT keypoints
	// number N of tilts to simulate t = 1, \sqrt{2}, (\sqrt{2})^2, ..., {\sqrt{2}}^(N-1)
	int num_of_tilts1 = 7;
	int num_of_tilts2 = 7;
	int verb = 0;
	// Define the SIFT parameters
	siftPar siftparameters;	
	default_sift_parameters(siftparameters);
	vector< vector< keypointslist > > keys1;		
	vector< vector< keypointslist > > keys2;		
	int num_keys1=0, num_keys2=0;	
	cout << "Computing keypoints on the two images..." << endl;
	time_t tstart, tend;	
	tstart = time(0);

	num_keys1 = compute_asift_keypoints(ipixels1_zoom, wS1, hS1, num_of_tilts1, verb, keys1, siftparameters);
	num_keys2 = compute_asift_keypoints(ipixels2_zoom, wS2, hS2, num_of_tilts2, verb, keys2, siftparameters);
	
	tend = time(0);
	cout << "Keypoints computation accomplished in " << difftime(tend, tstart) << " seconds." << endl;
	
	//// Match ASIFT keypoints
	int num_matchings;
	matchingslist matchings;	
	cout << "Matching the keypoints..." << endl;
	tstart = time(0);
	num_matchings = compute_asift_matches(num_of_tilts1, num_of_tilts2, wS1, hS1, wS2, 
										  hS2, verb, keys1, keys2, matchings, siftparameters);
	tend = time(0);
	cout << "Keypoints matching accomplished in " << difftime(tend, tstart) << " seconds." << endl;
	////////////Output image containing line matches (the two images are concatenated one above the other)
 int match_nums = (int)matchings.size();
 Mat resultshowv, resultshowh;
 vconcat(img1, img2, resultshowv);
 hconcat(img1, img2, resultshowh);
 matchingslist::iterator ptr = matchings.begin();
 Mat resultmatchpoints = Mat::zeros(Size(4, match_nums),CV_32F);
 vector<Point2f> src_pts, target_pts;
 for (int i = 0; i < match_nums; i++, ptr++) {
  Point v1_s1{ (int)(zoom1*ptr->first.x),(int)(zoom1*ptr->first.y) };
  Point v2_e1{ (int)(zoom1*ptr->second.x),(int)(zoom1*ptr->second.y + h1) };

  line(resultshowv, v1_s1, v2_e1 ,Scalar(0, 0, 255), 2);
  Point h1_s1{ (int)(zoom1*ptr->first.x),(int)(zoom1*ptr->first.y) };
  Point h2_e1{ (int)(zoom1*ptr->second.x +w1),(int)(zoom1*ptr->second.y) };
  line(resultshowh, h1_s1, h2_e1, Scalar(0, 0, 255), 2);
  resultmatchpoints.at<float>(i, 0) = v1_s1.x;
  resultmatchpoints.at<float>(i, 1) = v1_s1.y;
  resultmatchpoints.at<float>(i, 2) = v2_e1.x;
  resultmatchpoints.at<float>(i, 3) = h2_e1.y;
  src_pts.push_back({(zoom1*ptr->first.x),(zoom1*ptr->first.y)});
  target_pts.push_back({ (zoom1*ptr->second.x),(zoom1*ptr->second.y) });

 }
	//////////////////////////////////////////////////////////////////////////
 // 单映性矩阵
 Mat monomorphic_matrix = findHomography(src_pts, target_pts);
 //3.原图到目标图片的变换
 Mat im_out_perspective;
  warpPerspective(img1, im_out_perspective, monomorphic_matrix, img2.size());// 透视变换
 //
  Mat  im_out_affine;
  Mat rigid_matrix = estimateRigidTransform(src_pts, target_pts,true);
 warpAffine(img1, im_out_affine, rigid_matrix, img2.size());
	////////////////////////////////////////////////////////////////////////// 
	// Write all the keypoints (row, col, scale, orientation, desciptor (128 integers)) to 
	// the file argv[6] (so that the users can match the keypoints with their own matching algorithm if they wish to)
	// keypoints in the 1st image
	std::ofstream file_key1(argv[6]);
	if (file_key1.is_open())
	{
		// Follow the same convention of David Lowe: 
		// the first line contains the number of keypoints and the length of the desciptors (128)
		file_key1 << num_keys1 << "  " << VecLength << "  " << std::endl;
		for (int tt = 0; tt < (int) keys1.size(); tt++)
		{
			for (int rr = 0; rr < (int) keys1[tt].size(); rr++)
			{
				keypointslist::iterator ptr = keys1[tt][rr].begin();
				for(int i=0; i < (int) keys1[tt][rr].size(); i++, ptr++)	
				{
					file_key1 << zoom1*ptr->x << "  " << zoom1*ptr->y << "  " << zoom1*ptr->scale << "  " << ptr->angle;
					
					for (int ii = 0; ii < (int) VecLength; ii++)
					{
						file_key1 << "  " << ptr->vec[ii];
					}
					
					file_key1 << std::endl;
				}
			}	
		}
	}
	else 
	{
		std::cerr << "Unable to open the file keys1."; 
	}

	file_key1.close();
	
	////// keypoints in the 2nd image
	std::ofstream file_key2(argv[7]);
	if (file_key2.is_open())
	{
		// Follow the same convention of David Lowe: 
		// the first line contains the number of keypoints and the length of the desciptors (128)
		file_key2 << num_keys2 << "  " << VecLength << "  " << std::endl;
		for (int tt = 0; tt < (int) keys2.size(); tt++)
		{
			for (int rr = 0; rr < (int) keys2[tt].size(); rr++)
			{
				keypointslist::iterator ptr = keys2[tt][rr].begin();
				for(int i=0; i < (int) keys2[tt][rr].size(); i++, ptr++)	
				{
					file_key2 << zoom2*ptr->x << "  " << zoom2*ptr->y << "  " << zoom2*ptr->scale << "  " << ptr->angle;
					
					for (int ii = 0; ii < (int) VecLength; ii++)
					{
						file_key2 << "  " << ptr->vec[ii];
					}					
					file_key2 << std::endl;
				}
			}	
		}
	}
	else 
	{
		std::cerr << "Unable to open the file keys2."; 
	}
	file_key2.close();
	
    return 0;
}
