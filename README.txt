
Image Segment Project

Drive link for images: https://drive.google.com/drive/folders/1qelNLThIEIoukWZse1fXX7IWmHE-FvRM?usp=sharing

Overview
This code breaks down a picture into parts with similar colors and spots using Python and some math stuff.

Requirements
- Python
- Numpy
- OpenCV
- Matplotlib
- Os

Files
- segmentation.py: This is the code that does the breaking down thing.

How to Run
1. Put the image in the project folder.
2. Run segmentation.py.
3. It shows you different ways it broke down the picture.
4. It saves each version for you.

Process Overview
1. Opens your picture.
2. Figures out colors and spots.
3. Groups similar ones using K-means (a math method).
4. Splits the picture based on these groups.
5. Shows and saves the results.

Output
- Different regions of your image. They show the groups the code found.
