

#include "SuperPixels.h"



SuperPixels::SuperPixels() {


}

arma::mat SuperPixels::calculateSegmentation(cv::Mat& img_, int nSuperPixels, int display) {
   

    IplImage* img=new IplImage(img_);
	
	int NR_SUPERPIXELS=nSuperPixels;
    if ((!img))
      {
        printf("Error while opening file\n");
        
      }

    int width = img->width;
    int height = img->height;
    int sz = height*width;


    UINT* ubuff = new UINT[sz];
    //UINT* ubuff2 = new UINT[sz];
    //UINT* dbuff = new UINT[sz];
    UINT pValue;
    //UINT pdValue;
    char c;
    UINT r,g,b,d;
    int idx = 0;
    for(int j=0;j<img->height;j++)
      for(int i=0;i<img->width;i++)
        {
          if(img->nChannels == 3)
            {
              // image is assumed to have data in BGR order
              b = ((uchar*)(img->imageData + img->widthStep*(j)))[(i)*img->nChannels];
              g = ((uchar*)(img->imageData + img->widthStep*(j)))[(i)*img->nChannels+1];
              r = ((uchar*)(img->imageData + img->widthStep*(j)))[(i)*img->nChannels+2];
  			if (d < 128) d = 0;
              pValue = b | (g << 8) | (r << 16);
            }
          else if(img->nChannels == 1)
            {
              c = ((uchar*)(img->imageData + img->widthStep*(j)))[(i)*img->nChannels];
              pValue = c | (c << 8) | (c << 16);
            }
          else
            {
              printf("Unknown number of channels %d\n", img->nChannels);
          
            }
          ubuff[idx] = pValue;
          //ubuff2[idx] = pValue;
          idx++;
        }




  /*******************************************
   * SEEDS SUPERPIXELS                       *
   *******************************************/
  int NR_BINS = 5; // Number of bins in each histogram channel

  //printf("Generating SEEDS with %d superpixels\n", NR_SUPERPIXELS);
  SEEDS seeds(width, height, 3, NR_BINS);

  // SEEDS INITIALIZE
  int nr_superpixels = NR_SUPERPIXELS;

  // NOTE: the following values are defined for images from the BSD300 or BSD500 data set.
  // If the input image size differs from 480x320, the following values might no longer be
  // accurate.
  // For more info on how to select the superpixel sizes, please refer to README.TXT.
  int seed_width = 3; int seed_height = 4; int nr_levels = 4;
  if (width >= height)
  {
  	if (nr_superpixels == 600) {seed_width = 2; seed_height = 2; nr_levels = 4;}
  	if (nr_superpixels == 400) {seed_width = 3; seed_height = 2; nr_levels = 4;}
  	if (nr_superpixels == 266) {seed_width = 3; seed_height = 3; nr_levels = 4;}
  	if (nr_superpixels == 200) {seed_width = 3; seed_height = 4; nr_levels = 4;}
  	if (nr_superpixels == 150) {seed_width = 2; seed_height = 2; nr_levels = 5;}
  	if (nr_superpixels == 100) {seed_width = 3; seed_height = 2; nr_levels = 5;}
  	if (nr_superpixels == 50)  {seed_width = 3; seed_height = 4; nr_levels = 5;}
  	if (nr_superpixels == 25)  {seed_width = 3; seed_height = 2; nr_levels = 6;}
  	if (nr_superpixels == 17)  {seed_width = 3; seed_height = 3; nr_levels = 6;}
  	if (nr_superpixels == 12)  {seed_width = 3; seed_height = 4; nr_levels = 6;}
  	if (nr_superpixels == 9)  {seed_width = 2; seed_height = 2; nr_levels = 7;}
  	if (nr_superpixels == 6)  {seed_width = 3; seed_height = 2; nr_levels = 7;}
  }
  else
  {
  	if (nr_superpixels == 600) {seed_width = 2; seed_height = 2; nr_levels = 4;}
  	if (nr_superpixels == 400) {seed_width = 2; seed_height = 3; nr_levels = 4;}
  	if (nr_superpixels == 266) {seed_width = 3; seed_height = 3; nr_levels = 4;}
  	if (nr_superpixels == 200) {seed_width = 4; seed_height = 3; nr_levels = 4;}
  	if (nr_superpixels == 150) {seed_width = 2; seed_height = 2; nr_levels = 5;}
  	if (nr_superpixels == 100) {seed_width = 2; seed_height = 3; nr_levels = 5;}
  	if (nr_superpixels == 50)  {seed_width = 4; seed_height = 3; nr_levels = 5;}
  	if (nr_superpixels == 25)  {seed_width = 2; seed_height = 3; nr_levels = 6;}
  	if (nr_superpixels == 17)  {seed_width = 3; seed_height = 3; nr_levels = 6;}
  	if (nr_superpixels == 12)  {seed_width = 4; seed_height = 3; nr_levels = 6;}
  	if (nr_superpixels == 9)  {seed_width = 2; seed_height = 2; nr_levels = 7;}
  	if (nr_superpixels == 6)  {seed_width = 2; seed_height = 3; nr_levels = 7;}
  }

  seeds.initialize(seed_width, seed_height, nr_levels);



  //clock_t begin = clock();

  seeds.update_image_ycbcr(ubuff);

  seeds.iterate();

    
    
  //clock_t end = clock();
  //double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  //printf("    elapsed time=%lf sec\n", elapsed_secs);

  //printf("SEEDS produced %d labels\n", seeds.count_superpixels());
  
//  seeds.DrawContoursAroundSegments(ubuff, seeds.labels[nr_levels-1], width, height, 0xff0000, false);//0xff0000 draws red contours

    //     delete[] ubuff;
    //     delete[] output_buff;
 
  //std::string imageFileName = "./test_labels.png";
  //printf("Saving image %s\n",imageFileName.c_str());
  //seeds.SaveImage(ubuff, width, height,imageFileName.c_str());

  // DRAW SEEDS OUTPUT
    

  
//    arma::mat Label(img_.cols,img_.rows,arma::fill::zeros);
    
    arma::mat Label(img_.cols,img_.rows,arma::fill::zeros);
    for (int x=0; x<img_.cols;x++) {
        
        for (int y=0; y<img_.rows; y++) {
            //std::cout<<labels[y*image.cols+x]<<std::endl;//[y * width + x]
            Label(x,y)=seeds.get_labels()[y*img_.cols+x];
        }
        
    }
    
    if (display!=0) {
            seeds.DrawContoursAroundSegments(ubuff, seeds.labels[nr_levels-1], width, height, 0xff0000, false);//0xff0000 draws red contours
        cv::Mat c(img_.rows,img_.cols,CV_8U);
        
        for (int x=0; x<img_.cols; x++) {
            for (int y=0; y<img_.rows; y++) {
                c.at<uchar>(y,x)=ubuff[y*img_.cols+x];
            }
        }
        
        this->canvas=c;
    }
    
    


    
    
    
    delete img;
    delete[] ubuff;
    
    seeds.deinitialize();
    
   
    return Label;
    
  	// sz = 3*width*height;
  	//
  	//     UINT* output_buff = new UINT[sz];
  	//     for (int i = 0; i<sz; i++) output_buff[i] = 0;
  	//
  	//
  	//     //printf("Draw Contours Around Segments\n");
  	//     DrawContoursAroundSegments(ubuff, seeds.labels[nr_levels-1], width, height, 0xff0000, false);//0xff0000 draws red contours
  	//     DrawContoursAroundSegments(output_buff, seeds.labels[nr_levels-1], width, height, 0xffffff, true);//0xff0000 draws white contours
  	//
  	//    	std::string imageFileName="";
  	//     imageFileName = "./test_labels.png";
  	//     //printf("Saving image %s\n",imageFileName.c_str());
  	//     SaveImage(ubuff, width, height,
  	//               imageFileName.c_str());
  	//
  	//        imageFileName = "./test_boundary.png";
  	//     //printf("Saving image %s\n",imageFileName.c_str());
  	//     SaveImage(output_buff, width, height,
  	//               imageFileName.c_str());
  	//
  	//
  	//     std::string labelFileNameTxt = "./test_.seg";
  	//     seeds.SaveLabels_Text(labelFileNameTxt);
  	//
  	//
  	//

}