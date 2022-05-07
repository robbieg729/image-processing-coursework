## Coursework outline
Implementing denoising, dewarping, and contrast enhancement to a set of artificially corrupt images to improve YOLO object detection performance.
Corrupt images are contained in the "images" directory, with 2 sub-folders, namely "test" and "validation". The "test" folder contains a set of artificially
corrupt images, as well as the results of the image processing techniques on those images. The "validation" folder contains a different set of corrupt images,
the results on those images, and the ground truth (i.e. the original, uncorrupt images).

Code implementation exists in the file "processing.py". 2 additional Python files were provided to us, namely "compare_images.py", which compares 2 sets of images based on various metrics such as MSE, and "yolo.py", which creates a video file implementing YOLO object detection on a set of images. Note an additional file was required to successfully use YOLO, but its size was too large to be uploaded. Full brief of the assignment can be found in "Instructions.pdf", and a report detailing and explaining
the code implementation can be found in "Report.pdf". The "videos" directory contains YOLO videos on several of the image sets in "images".

## Coursework result
Achieved 80 out of a possible 100 marks, yielding 80% as my final mark.