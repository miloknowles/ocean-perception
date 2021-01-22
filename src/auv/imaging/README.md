## Converting RAW Sea-thru Images

Dataset: http://csms.haifa.ac.il/profiles/tTreibitz/datasets/sea_thru/index.html

```bash
OpenCV had trouble opening the file types that come with Sea-thru. For example, the NEF images were 1/10th the correct size when loaded with cv::imread.

NOTE: This conversion is really slow! Seems like working with RAW images is just slow in general.

# For D5
mogrify -format png *.NEF
# This also might work: ufraw-batch --curve=linear --out-type=png *.NEF

# For D3
# http://manpages.ubuntu.com/manpages/bionic/man1/ufraw-batch.1.html
sudo apt install ufraw
ufraw-batch --curve=linear --out-type=png *.ARW
```
