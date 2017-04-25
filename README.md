# Image Processing Tips

I just finished term 1 of the Udacity self-driving car course. Term 1 has five projects and all of them required some form of image processing (to read, process and display images) as a pre-processing step for computer vision and/or deep learning tasks. The key to get better results for these tasks is to get the image processing done accurately. Often times this involves proper scaling between JPEG and PNG image formats or converting to grayscale and normalizing an image or simply using image processing packages OpenCV and Matplotlib correctly for the task at hand. In the following I address all these topics and provide recommendations. The projects mentioned in this write-up refer to Udacity term 1 projects. However, the following post does not require any knowledge of these projects.

# When to use OpenCV and Matplotlib?

Both OpenCV and Matplotlib can be used to read and display images. OpenCV reads and plots an image in the BGR format while Matplotlib reads and plots in the RGB format. I like to use Matplotlib to read and display images as 1) it is convenient to plot inside a Jupyter notebook and 2) by reading in Matplotlib I don’t have to convert to a different color space for display. However, if I am processing both PNG and JPEG image formats, I use OpenCV (reasons discussed later). Here is some code to illustrate these points (Check my github repository for [example code](https://github.com/kharikri/ImageProcessingTips/blob/master/Image%20Processing%20Tips%20Example%20Code.ipynb)):

* To read and display images in OpenCV:

        import cv2

        image = cv2.imread(‘images/test.jpg’)
        cv2.imshow( “Display window”, image )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

* To read and display images with Matplotlib:

        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        %matplotlib inline

        image = mpimg.imread(‘images/test.jpg’)
        plt.imshow(image)

* Displaying in OpenCV opens another window. With Matplotlib we can **display inline inside a Jupyter notebook** which is very convenient
* OpenCV reads and plots an image in the BGR format while Matplotlib reads and plots in the RGB format. Hence if we read an image with OpenCV and to display with Matplotlib, we need to convert it from BGR to RGB format as follows:

        imageBGR = cv2.imread(‘images/test5.jpg’)
        imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
        plt.imshow(imageRGB)
* **To avoid this color conversion I use Matplotlib for reading and displaying images which keeps everything in the RGB color space**. We can still use OpenCV for image processing functions as it supports direct conversion from RGB to any other color space with the `cvtColor()` command
* As a side note, for grayscale images there is no issue reading them with OpenCV and displaying with Matplotlib as there is only one color channel and conversion is not needed:

        image = cv2.imread(‘images/charlie_grayscale.jpg’)
        plt.imshow(image, cmap = ‘gray’)

# Why convert to grayscale from color?

Many image processing and computer vision algorithms (Canny, Hough, Sobel) use grayscale images rather than color images. These tasks involve edge detection and color information is not useful, hence grayscale is just fine. Also grayscale processing is at least three times faster than that of color image processing. This is because grayscale image has only one color channel as opposed to three in a color image. I have shown an example of this time savings in the example code.

# Why normalize data before training?

When we normalize data we typically make the data have zero mean and unit variance with a formula such as:
![alt text](https://github.com/kharikri/ImageProcessingTips/blob/master/Images/NormalizationFormula.png)
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  For example, for a grayscale image Xmin is 0, Xmax is 255 and Xnorm is between 0 and 1.

We get two benefits with normalization. First, if data is not normalized, features with larger numerical values dominate features with smaller numerical values and consequently we will not get contributions from features with smaller values. In Project 5 — Vehicle Detection and Tracking, if we are extracting three different kinds of features (HOG, Spatial binning, and Color transforms) it is an absolute must to normalize them. Otherwise larger feature values will dominate smaller feature values.

Second, many learning algorithms behave well with normalized data. This manifests in higher test accuracy for normalized data than with non-normalized data. We can easily check this with Project 2 on Traffic Classification.

# How to deal with PNG and JPEG images?

We will have situations where we will have training data in PNG format and test data in JPEG format ([Project 5 — Vehicle Detection and Tracking](https://github.com/kharikri/SelfDrivingCar-VehicleDetectionAndTracking)) or the reverse. As mentioned before, in such situations I prefer reading the image data in OpenCV instead of Matplotlib. The reason being OpenCV reads both PNG and JPG in 0 to 255 range while Matplotlib reads JPEG in 0 to 255 and PNG on 0 to 1 range! With Matplotlib if we do not scale the image data appropriately we will get strange results.

To illustrate this I’ll borrow Project 5 as an example. Here I have trained a Support Vector Machine (SVM) with **PNG** files which were read with **Matplotlib**. Using these SVM parameters I detect/predict cars in a **JPEG** test image and draw bounding boxes around them. The test image is shown below:

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
![alt text](https://github.com/kharikri/ImageProcessingTips/blob/master/Images/OriginalImage.png)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Test Image

If we do not account for scaling we get the following result. This incorrect result is because Matplotlib reads PNG in 0 to 1 range and JPEG in 0 to 255 and we need to scale appropriately.

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
![alt text](https://github.com/kharikri/ImageProcessingTips/blob/master/Images/BBUnscaledImage.png)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Bounding boxes on unscaled Image

To scale the JPEG image divide the test image data by 255 with the following line of code:

        image= image.astype(np.float32)/255

This result is shown in the following picture:

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
![alt text](https://github.com/kharikri/ImageProcessingTips/blob/master/Images/BBScaledImage.png)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Bounding boxes on scaled Image

Hence to avoid these scaling issues I simply use OpenCV when I have to process both PNG and JPEG image formats as it reads these two image formats in the same (0 to 255) range.

# Conclusion

These image processing tips mentioned above in this post are straightforward and easy to follow and will save time and frustration when debugging computer vision and deep learning algorithms.
