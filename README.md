# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate

This is a collection of all labs and homework done in my CSCI-509 - AI & Computer Vision class. I do not claim to own any of the images being used as examples for educational purposes and testing.

**Everything in this README file will be directed at a Windows 10/11 environment. I will not be showing linux commands necessary to get this working.**

NOTE: Some applications installed or some steps taken might not be needed. But I don't remember specifically what I did on Windows 10/11 to setup my environment for AI, even though I have done it multiple times as I factory reset my PC and laptop every few months. I will update this README in the future then next time I factory reset my PC or laptop to reflect the proper steps I go through.

---

# Required Project Tools

- [Visual Studio Code](https://code.visualstudio.com/)
   - Not needed but highly recommended. This should be the only editor anyone ever uses. Better than PyCharm, IntelliJ, Atom, etc. by a million miles (at least for my uses so far in life).
- [Anaconda3](https://www.anaconda.com/)
   - Definitely needed.
   - Be sure to select "Add Anaconda3 to the system PATH environment variable" when installing. This is extremely useful.
- [CMake](https://cmake.org/download/)
   - Definitely needed.
- [MSYS2](https://www.msys2.org/)
   - Might not be needed.
   - Be sure to follow the given installation steps on that site.

---

# Environment Setup

The instructions below are directly from my teacher. I will change/update if or when necessary.

- Open Command Prompt and copy/paste each line one by one into your terminal shell (active directory doesn't matter for setup):
   - conda create -n Summer2022 python=3.6.6
      - NOTE: Environment name doesn't have to be Summer2022. Name it whatever you want. Just remember when you see "Summer2022" you should enter your environment name.
   - conda activate 2022
   - pip install --upgrade pip
   - pip install opencv-contrib-python==4.0.1.24 numpy==1.19.5 pyyaml matplotlib pillow imageio tqdm six
   - pip install tensorflow==1.12.0
   - conda remove keras -force
   - pip install keras==2.2.4 --no-deps
   - pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f
      - For me, just doing "pip install dlib" did not work. The only way to install dlib is to run the command with the link above.
   - conda deactive

---

# Working with the Project and Running Code

- Open Command Prompt and navigate to /AI-and-Computer-Vision-Class/Student_folder/.
   - For me, the command is:  **cd "C:/GitHub/AI-and-Computer-Vision-Class/Student_folder"**
- Activate the Anaconda environment.
   - **conda activate Summer2022**
- Running one of the python files.
   - EX: **python BlurringImage.py**

