# University_Id_Card_Detector
## Description
This is a Simple project that have been developed for extracting details from Uva Wellassa University Id Card. We created this for our third year Image Processing Project. In here if you provide a valid Id card then it extract the details and save it to file. **Note**: This Project is still under development. So if you like to contibute you can contribute to this project.

## Setup
1. install Tensorflow [setup instructions](https://www.tensorflow.org/install)
2. install opencv [setup instructions](https://pypi.org/project/opencv-python/), [Docs](https://opencv.org/)
3. install pytesseract [setup instructions](https://pypi.org/project/pytesseract/)
4. install Streamlit [setup instructions](https://docs.streamlit.io/en/stable/)
```
pip install tensorflow
pip install cv2
pip install pytesseract
pip install streamlit
```

## Methodology 
In here first I have trained a model using Neural Network to detect the Id Card and the did some opencv methods to extract your photo in Id card. and used Teserect-ocr LSTM to extract the text in image.
## Future Work
1. We need to extract only the Name,ID,Faculty and Degree Part from the whole Image insted of extracting all the text
2. We need to add only the id card (Document Scanned ID) to get a better and perfect result. So we need to find a way to solve it.

## Screen Shots
![GitHub](https://github.com/farookfazni/University_Id_Card_Detector/blob/master/1.PNG?raw=true)


![GitHub](https://github.com/farookfazni/University_Id_Card_Detector/blob/master/2.PNG?raw=true)


![GitHub](https://github.com/farookfazni/University_Id_Card_Detector/blob/master/3.PNG?raw=true)
