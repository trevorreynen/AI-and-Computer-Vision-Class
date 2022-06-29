# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate

This is a collection of all labs and homework done in my CSCI-509 - AI & Computer Vision class. I do not claim to own any of the images/videos/etc. being used as examples for educational purposes and testing.

**Everything in this README file will be directed at a Windows 10/11 environment.**

NOTE: Some applications installed or some steps taken might not be needed. I don't remember specifically what I did on Windows 10/11 to set up my environment for AI, even though I have done it multiple times as I factory reset my PC and laptop every few months. I will update this README in the future then next time I factory reset my PC/laptop to reflect the proper steps I go through.

---

## Required Project Tools

- [Visual Studio Code](https://code.visualstudio.com/)
   - Not needed but highly recommended. This should be the only editor anyone ever uses. Better than PyCharm, IntelliJ, Atom, etc. by a million miles (at least for my uses so far in life).
- [Anaconda3](https://www.anaconda.com/)
   - Required.
   - Be sure to select "Add Anaconda3 to the system PATH environment variable" when installing. This is extremely useful.
- [CMake](https://cmake.org/download/)
   - Might not be needed.
- [MSYS2](https://www.msys2.org/)
   - Might not be needed.
   - Be sure to follow the given install steps on that site.

---

## Environment Setup

The instructions below are directly from my teacher. I will change/update if or when necessary.

- Open Command Prompt and copy/paste each line one by one into your terminal shell (active directory doesn't matter for setup):
   - conda create -n Summer2022 python=3.6.6
      - NOTE: Environment name doesn't have to be Summer2022. Name it whatever you want. Just remember when you see "Summer2022" you should enter your environment name.
   - conda activate Summer2022
   - pip install --upgrade pip
   - pip install opencv-contrib-python==4.0.1.24 numpy==1.19.5 pyyaml matplotlib pillow imageio tqdm six imutils
   - pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f
      - For me, just doing "pip install dlib" did not work. The only way to install dlib is to run the command with the link above.
   - pip install tensorflow==1.12.0
   - pip install keras==2.2.5
   - conda deactivate

---

## Working with the Project and Running Code

- Open Command Prompt and navigate to ./AI-and-Computer-Vision-Class/Student_folder/.
   - For me, the command is:  **cd "C:/GitHub/AI-and-Computer-Vision-Class/Student_folder"**
- Activate the Anaconda environment.
   - **conda activate Summer2022**
- Running one of the python files.
   - EX: **python BlurringImage.py**

---

## The final two labs and the ./Student_folder/research/ directory - IMPORTANT INFO

In order to run the last two labs of my class, this research folder with the object_detection/ folder and setup.py inside is needed. It is used to create the object_detection module in our Virtual Environment, allowing the ability to run the necessary code.

Any files and folders inside the ./Student_folder/research/ directory are a small portion of what I received from my teacher. However, I do know that he originally got them from the [TensorFlow Model Garden](https://github.com/tensorflow/models). So, if you were interested in what's missing or the subject, check that link out instead.

***I deleted as much as I could from what my teacher gave me in order to save space AND maintain the ability to create the object_detection module required to run the ObjectDetection programs*** in the ./Student_folder/ directory. I still need to test this fact to make sure, but it should work fine. Follow the steps below.

### Steps to create the object_detection module (only need to do this once):
1. Open Command Prompt and navigate to ./AI-and-Computer-Vision-Class/Student_folder/research/.
   - For me, the command is:  **cd "C:/GitHub/AI-and-Computer-Vision-Class/Student_folder/research"**
2. Activate the Anaconda environment.
   - **conda activate Summer2022**
3. Install Protobuf
   - **conda install -c anaconda protobuf**
   - NOTE: I need to test this, but I am not sure if that command alone allows the use of protoc or if something else is needed to be installed on the side. Will find out soon and change info as needed.
4. Run the following Protobuf command within the research folder:
   - **protoc object_detection/protos/*.proto --python_out=.**
5. Run the following pip command, still within the research folder and conda env activated:
   - **python -m pip install . --use-feature=in-tree-build**

At this point, you will no longer need to bother with anything inside the ./Student_folder/research/ directory. Just ignore the research folder. You should now be able to go back to the Student_folder directory and run ObjectDetection_Image.py and ObjectDetection_Live.py without problems.

---

# One Last Note

The model needed for the labs **FaceRecognition.py** and **FaceRecognition_Matching.py**, called "**vgg_face_weights.h5**" is over 100 MB. For now, I will not be uploading it to GitHub inside this repository. If I find a link to the model I will either add it here or add code to the two FaceRecognition programs to download the model if it isn't already.

