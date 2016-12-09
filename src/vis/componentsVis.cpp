/*
 * componentsVis.cpp
 *
 *  Created on: Dec 4, 2015
 *      Author: Michal Busta at gmail.com
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "componentsVis.h"

using namespace cv;

namespace cmp{

static void drawCharacter(const LetterCandidate& region, Mat& output, CvMemStorage* storage, Mat& green, Mat& red, cv::Mat& blue )
{
	if (region.bbox.br().x >= output.cols || region.bbox.br().y >= output.rows)
		return;


  static int mask_id;
  std::cout << "dbg>drawCharacter>mask " << mask_id << ", region.bbox=" << region.bbox << ", scaleFactor=" << region.scaleFactor << ", mask.rows=" << region.mask.rows << ", cols=" << region.mask.cols << std::endl;
  imwrite("/tmp/mask"+std::to_string(mask_id)+"_origin.png", region.mask);

	Mat maskImage =  region.mask;
	if( region.scaleFactor != 1.0)
	{
		cv::Mat scaledMask;
		cv::resize(maskImage, scaledMask, cv::Size(roundf(maskImage.cols * region.scaleFactor), roundf(maskImage.rows * region.scaleFactor)));
		maskImage = scaledMask;
    imwrite("/tmp/mask"+std::to_string(mask_id)+"_scaled.png", scaledMask);
	}else{
		maskImage =  region.mask.clone();
	}

	IplImage iplContours = maskImage;
	CvSeq *contour;
	cvFindContours( &iplContours, storage, &contour, sizeof(CvContour),
			CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0) );


  //dbg
  vector<vector<Point> > dbg_contours;
  vector<Vec4i> dbg_hierarchy;
  findContours( maskImage, dbg_contours, dbg_hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
  Mat drawing;
  cvtColor(maskImage, drawing, CV_GRAY2BGR);
  RNG rng(12345);
  for( int i = 0; i< dbg_contours.size(); i++ ) {
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    drawContours( drawing, dbg_contours, i, color, 2, 8, dbg_hierarchy, 0, Point() );
  }
  imwrite("/tmp/mask"+std::to_string(mask_id)+"_contours.png", drawing);


	cv::Rect roid = region.bbox;
	roid.width += 1;
	roid.height += 1;
	Mat roi = output(roid);
  imwrite("/tmp/mask"+std::to_string(mask_id)+"_roi.png", roi);
	try{
	cv::Mat iplMask = output(region.bbox);
  imwrite("/tmp/mask"+std::to_string(mask_id)+"_bbox.png", iplMask);
  std::cout << "dbg>drawCharacter>region.isWord=" << region.isWord << ", quality=" << region.quality << std::endl;
	if( !region.isWord )
	{
		if( region.quality > 0.5 )
			cv::add((1 - region.quality) * iplMask, region.quality * 0.3 * green(region.bbox), iplMask, maskImage);
		else
			cv::add((region.quality) * iplMask, (1 - region.quality) * 0.3 * blue(region.bbox), iplMask, maskImage);
	}else{
		cv::add((1 - region.quality) * iplMask, region.quality * 0.3 * red(region.bbox), iplMask, maskImage);
	}
	}catch(...){
		std::cout << "Roi: " << region.bbox << ", cols: " << maskImage.cols << ", rows: " << maskImage.rows << std::endl;
	}
	//cvDrawContours( &iplMask, contour, color, CvScalar(), 1);

  imwrite("/tmp/output"+std::to_string(mask_id)+"_diff.png", output);
  mask_id++;
}


Mat createCSERImage(std::vector<LetterCandidate*>& regions, const std::vector<cmp::FastKeyPoint>& keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, const Mat& sourceImage)
{
	Mat greyImage;
	if(sourceImage.channels() == 3)
	{
		cvtColor(sourceImage, greyImage, CV_RGB2GRAY);
	}else{
		greyImage = sourceImage;
	}
	sort(regions.begin(), regions.end(),
	    [](const LetterCandidate * a, const LetterCandidate * b) -> bool
	{
		if( a->isWord != b->isWord )
		{
			if( !a->isWord )
				return false;
			else
				return true;
		}
	    return a->quality < b->quality;
	});



	Mat output;
	cvtColor(greyImage, output, CV_GRAY2RGB);

	RNG rng(12345);
	for (vector<LetterCandidate*>::const_iterator j = regions.begin(); j < regions.end(); j++)
	{
		Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
		for(auto kpid : (*j)->keypointIds)
		{
			const cmp::FastKeyPoint& kp = keypoints[kpid];
			if(kp.octave != (*j)->keyPoint.octave)
				continue;
			std::pair <std::unordered_multimap<int,std::pair<int, int> >::iterator, std::unordered_multimap<int,std::pair<int, int>>::iterator> ret;
			ret = keypointsPixels.equal_range(kp.class_id);
			//if( keypointsPixels.size() > 0)
			//	assert( std::distance(ret.first, ret.second) > 0 );
			for (std::unordered_multimap<int,std::pair<int, int> >::iterator it=ret.first; it!=ret.second; ++it)
			{
				cv::circle(output, cv::Point(it->second.first * (*j)->scaleFactor, it->second.second * (*j)->scaleFactor), 1 * (*j)->scaleFactor, color);
			}
		}
	}

  //dbg
  Mat cand = output.clone();
  for(int i = 0; i < regions.size(); i++){
		Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
    cv::rectangle(cand, regions[i]->bbox.tl(), regions[i]->bbox.br(), color);
  }
  imwrite("/tmp/output_candidate.png", cand);


	CvMemStorage* storage = cvCreateMemStorage();
	Mat green = Mat(sourceImage.size(), CV_8UC3, CV_RGB(0, 255, 0));
	Mat red = Mat(sourceImage.size(), CV_8UC3, CV_RGB(0, 0, 255));
	Mat blue = Mat(sourceImage.size(), CV_8UC3, CV_RGB(255, 0, 0));
	for (vector<LetterCandidate*>::const_iterator j = regions.begin(); j < regions.end(); j++)
	{
		if( (*j)->quality < 0.1 )
			continue;

		if( (*j)->duplicate != -1 )
			continue;

		//std::cout << "q: " << (*j)->quality << ", " << (*j)->isWord << std::endl;
		drawCharacter(**j, output, storage, green, red, blue);
	}


	cvReleaseMemStorage(&storage);

	return output;

}

}//namespace cmp
