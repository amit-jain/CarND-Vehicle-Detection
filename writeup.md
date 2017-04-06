
**Vehicle Detection Project**
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/car_hog.png
[image3]: ./output_images/heat_map.png
[video1]: ./project_video_tracked.mp4
[video2]: ./project_video_lane_tracked.mp4
[video3]: ./test_video_tracked.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  
The project has been uploaded to my fork [here](https://github.com/amit-jain/CarND-Vehicle-Detection).  

My project includes the following files:
- `vehicle_tracking.ipynb`
 jupyter notebook with code for the processing pipeline.
- `writeup.md`
 report summarizing the results
- `output_images` folder having the output images

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the sixth code cell of the jupyter notebook `vehicle_tracking.pynb`).  

I started by reading in all the `vehicle` and `non-vehicle` images. Here is an example of one of each of the 
`vehicle` and `non-vehicle` classes:

![vehicle and non-vechicles][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, 
and `cells_per_block`).  I grabbed top three images from each of the two classes and displayed them to get a feel for 
what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and
 `cells_per_block=(2, 2)`:


![Hog images][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and decided on the following parameters:
```
* color_space = 'YCrCb'
* orient = 9
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = 'ALL'
```

These parameters were initially arrived at after hit and trial for the best classification error. But these finalized
 parameters were ultimately driven by testing on the project video. The other various combinations tried was using `YUV` color_space and using `12` orientations but they seemed to induce more false positives. The HOG features looked useful for all the three color channels (as seen in the image above) and hence ALL was used for hog_channel. 


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using sklearn.svm.LinearSVC. Only the subset of KITTI and GTI data provided was used for training. The data consists of 8792 car images and 8968 non car images. The data was split into 80% training and 20% test set. The training code is in the block 220 of the jupyter notebook.

The parameters finalized are:
```
color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (16, 16)
hist_bins = 16
spatial_feat = True
hist_feat = True
hog_feat = True
```
These parameters were primarily driven by the testing on the project video to maximize the accuracy and minimize false positives. The hog parameters chosen are described above. In addition I used histogram with 16 bins as well as spatial binning with dimensions (16, 16) to train the classifier. I experimented with 32 bins and (32, 32) bins respectively for histogram and spatial binning respectively but they seemed to induce more false positives when the the classifier was run on the video. Similarly, experimenting with 12 orientation for hog features induced more false positives.

The feature vectors are normalized via scikit-learn's StandardScaler.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the Hog sub-sampling sliding window for search as described in the lesson. The meat of the code is in the `find_cars` function in block 221 of the notebook. The default window size used is 64 (8 * 8) and the step size is 2 cells which introduces a 75 % overlap.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately the search was done on 6 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice balance between removal of false positives and vehicle detection.
 Also, the threshold for heat map was set to 5. The image size was trimmed to search only within the height of (400, 656).
 
 The finalization of the parameters was all driven by experimentation on the project video where the main problem was reducing the detection of vehicles on the opposite lanes and detection of vehicle further away which appear smaller in the actual lanes. 
 After a lot of experimentation there were 6 scales chosen. There are 3 scales `1.27`, `1.28`, `1.29` trying to concentrate on top part of the image within the heights (388, 520) while the other 3 scales `1.75`, `1.9` and `2` focus on the entire image of interest. The scales were chosen to be almost redundant to have more heat concentrated on the actual vehicles of interest and very low heat if at all on the vehicles not of interest as well as on random false positive detections. Then using a high threshold the false positives could be filtered out.
 
 The positions of positive detections in each frame of the video was recorded.  From the positive detections a heatmap was created and then thresholded that map to identify vehicle positions.  After this `scipy.ndimage.measurements.label()` was used to identify individual blobs in the heatmap with the assumption that each blob identified corresponded to a vehicle.  Then bounding boxes were constructed to cover the area of each blob detected.
   
 Here are the results after applying the pipeline described above to each of the test images (including an image from the video which proved to be a challenge for vehicle detection):

![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]

Integrated passively (not using any information from) the lane finding code from the previous project to have simultaneous lane finding and vehicle detection. The code used is reproduced from the previous project and is in block 227, 228.
 Here's a [link to integrated video result][video2]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The mechanism to search for and threshold is described above in detail. In addition to searching and thresholding that was done per frame there were 2 things which were done to have a smoother bounding box as well as smoother detections for a series of frames.
The code is in block 224 of the jupyter notebook. 
* The bounded boxes detected were smoothed over multiple frames and then thresholded again to eliminate the false positives and remove vehicles detected in only very few of the frames, which were almost all the vehicles from the opposite lane.
* `Vehicle` class which smoothed over the bounding boxes per vehicles already detected. Then these bounding boxes were again used to identify individual blobs using `scipy.ndimage.measurements.label()`. This led to a smoother detection of the vehicles. This was mostly done to reduce multiple bounding boxes on the same vehicle being detected as the heat and the label mechanism described above in the sliding window search sometimes identified blobs over the same car separately.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* _Efficieny_ -
    One of the problems (already identified above) was to eliminate vehicles from opposite direction as the smoothing done would prolong the detection much beyond the few frames that actual vehicle was detected leading to many false positives. But reducing this by increasing thresholding and/or using less scales for searching reduced detection of desired vehicles further away which appear smaller. While the project was able to effectively work around the problem with multiple scales it is less efficient. At about 1.6 s/frame it may not be too useful in a real car. 
    To have a proper solution it might be needed to use the lane tracking solution to only limit the space for searching in positional proximity to the lane. 
    * Related to above the solution is less generic and would likely have false positives on other videos.
* _Classification accuracy_ - 
    Though it did not seem that the classifier was a problem as it was detecting vehicles fairly well sometimes, even the fuzzy cars going in the opposite direction but more data like the Udacity open source data should be considered and trained on to have a more generic solution. It would also lead to lesser false positives implicitly rather than having to remove them using techniques like thresholding, multiple scales etc.
    * A particular appealing idea (related to the above) is to use deep learning for classifying which could also lead to better accuracy.
 

