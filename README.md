# Math Expression Recognition

_Math Expression Recognition_ is an implementation on recognizing the characters 
and math symbols from a image that is a sreenshot of a digital document or a photo
of a printed document containing math expression, and reconstructing them to generate
the corresponding _LaTeX_ expression which is useful in document editing and 
automation. Its image preprocessing and character predicting part is base on 
[openCV](https://opencv.org/). It's extensible to work on more math fonts.

## Using the executable file
To use the compiled executable file of this project, download the **Win pack** of 
openCV from [openCV Releases](https://opencv.org/releases.html), extract it, and 
locate the `opencv_world341d.dll` (pay attention to the **`d`** standing for debug) 
inside the extracted files: "`.../opencv/build/x64/vc15/bin/`". 
Then copy the dll file to the same folder where the executable file located and 
it's ready to be executed.

## Using the source file
To use and work with the source file `MathExpressionRecognition.cpp`, first setup a
project in Microsoft Visual Studio that works with openCV referring to this 
[tutorial](https://docs.opencv.org/3.4.1/dd/d6e/tutorial_windows_visual_studio_opencv.html), 
or refer to other similar tutorials if other IDE is used. 

Notice that a copy of the 
openCV's dll dynamic library file is also required under the same folder with the 
source file. Recent releases of openCV do not include complete static library, so 
working with dll library is more effortless or it's necessarily to compile 
the openCV library.

---

## Establishing data set
If image of printed math expression in other fonts instead of the default one is to 
be recognized, then a standard data set of the font type is required. Basic 
workflow: 
1. Take sreenshot of each character or symbol with a white background without any 
other useless information in the background. Refer to the original bundle of images
for the basic appearance of the standard images.

	> Currently characters or symbols with 
only a single separated part is supported except '=', 'i', 'j' and a few others 
that have been taken into consideration. 

	Higher resolution of the standard images is recommended.

2. Rename the images in specific order and put them all in one folder. 

	* If the source code is not meant to be modified, then the same characters and 
	symbols should have the same file name (File type name can be different but 
	all the images should be in the same file type) as ones in the default bundle 
	of images. 

	* If the images are in different order, source code must be adapted to the 
	different lables mainly in `correspondSymbol(...)` and `mathString(...)` case 
	switching expressions.

	* If there are images of extra characters and symbols comparing to the default 
	one, the algorithm of the reconstructing part has to be modified.

3. Establish the binary data set in the executable

	Follow the prompt in the program to establish the binary data set of the 
	standard images and their corresponding lables. **The images and lables dataset bundled with this instruction can be utilized as an example.**

---

## Information
Feel free to improve this program mainly focusing on better image preprocessing, 
more accurate lable predicting and more general structure 
reconstructing.
