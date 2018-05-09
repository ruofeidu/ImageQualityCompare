# C++ Implementation to Compare Pairs of Image Quality using RMSE, SSIM, and PSNR
Compare the quality between two images using RMSE, SSIM, and PSNR.
The values of the PSNR can be predicted from the SSIM and vice-versa. The
PSNR and the SSIM mainly differ on their degree of sensitivity to image degradations. 

## Usage
The executable file is under x64/Release/Compare.exe
```bash
Compare image_file_name_1 image_file_name_2 [--mask] [--block_size] 2
```
* The optional mask parameter will neglect the total white or black pixels in the image1.
* The optional block_size parameter determines the window size of SSIM.
* The smaller block_size is, the slower it runs.

## Compilation
If the compilation fails, please fix the following environment variables:
* OPENCV_INC: Directory to OpenCV include folder.
* OPENCV_LIB: Directory to OpenCV libraries.
* PATH: Add [the executable DLLs of OpenCV and GLUT](https://obj.umiacs.umd.edu/dll/DuEngineLibs.zip) into an arbitrary directory of PATH.

# Details
## RMSE: Root Mean Squared Error
The square root of the mean squared error. More sensitive than SSIM and PSNR in values. 
Wiki: https://en.wikipedia.org/wiki/Root-mean-square_deviation

## SSIM: Structural Similarity Index Measure
The SSIM index is calculated on various windows of an image. 
Wiki: https://en.wikipedia.org/wiki/Structural_similarity

## PSNR: Peak signal-to-noise ratio
PSNR is most often defined via 10 multiplies the log10 of the normalized mean squared error (MSE).
Wiki: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

# Free software!
## Contact: Ruofei Du (http://www.duruofei.com)
