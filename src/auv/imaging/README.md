## Notes on the RAW Sea-Thru Images

I've had trouble getting the downloaded Sea-thru images to look like Figure 6 in the [CVPR paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.pdf).

Viewing .ARW files on Ubuntu using ufraw:
- Open the image with ufraw
- Go to the "Color management" pane. Set Gamma=1.00 and Linearity=0.0. This ensures that the linear (not gamma corrected) intensities are being viewed. Linear images are required for processing with the Sea-thru algorithm.
- Go to the "White balance" pane (1st one). The "Camera WB" option results in a blue-green tint. Maybe this is realistic?
- Manually setting R=4.4, G=1.0, B=1.6, EV=3.0 gives a neutral tint. But maybe this is just removing the underwater light color?

## Converting RAW Sea-thru Images

Dataset: http://csms.haifa.ac.il/profiles/tTreibitz/datasets/sea_thru/index.html

```bash
OpenCV had trouble opening the file types that come with Sea-thru. For example, the NEF images were 1/10th the correct size when loaded with cv::imread.

NOTE: This conversion is really slow! The .ARW files are pretty huge.

# For D5
mogrify -format png *.NEF
# This also might work: ufraw-batch --curve=linear --out-type=png *.NEF

# For D3
# http://manpages.ubuntu.com/manpages/bionic/man1/ufraw-batch.1.html
sudo apt install ufraw
ufraw-batch --curve=linear --out-type=png *.ARW

# Gives the most normal looking colors
# These images should have a neutral tint
ufraw-batch --temperature=23000 --green=0.323 --gamma=1.0 --linearity=0.0 --exposure=3.0 --base-curve=linear --curve=linear --out-type=png --shrink=8 *.ARW

# Uses ufraw default "camera" color transformation
# These images will have a blue-green tint
ufraw-batch --wb=camera --gamma=1.0 --linearity=0.0 --exposure=3.0 --base-curve=linear --curve=linear --out-type=png --shrink=8 *.ARW
```
/
