# PET(Physical Education Tool)
## Inspiration
The four of us on the team experienced a full year of virtual learning during our sophomore of high school. Perhaps the class that was impacted the most was Physical Education or PE. It was extremely difficult for the teacher to monitor students’ form during various exercises, especially when there were 30 videos playing simultaneously. As an alternate method for feedback, our teachers requested video submissions and provided personal comments. We realized that this process was very tedious for teachers and made it difficult for students to improve their form for difficult exercises. As a result, we thought of PET (Physical Education Tool), a method to automate the feedback a teacher would normally provide to a student.

## What it does
PET is a software that can be used to help physical education teachers ensure their students are performing various exercises correctly through a virtual environment. The user uploads a video of them doing a squat, curl up, bicep curl, plank, and/or push up to the website, and PET provides real-time feedback of what, if anything, the user needs to improve about their form.

## How we built it
We developed the machine learning and computer vision algorithms using Google's MediaPipe and OpenCV.  The UI was created using Flask. We developed code using Google Colaboratory and Visual Studio Code before integrating individual pieces into the overall system using GitHub.

## Challenges we ran into
For many of our exercises, we had to find out when the person was at the end of their rep (e.g. bottom of pushup, top of curl up) so that we knew the specific frame in the video to critique. We had to analyze each frame with respect to its surrounding frames in the video to determine what exactly the student was doing. 

On the web development portion, we wanted to provide as much flexibility to the user so our plan was to allow video uploads and live videos taken directly on the website. At first, we were only able to get live videos to be compatible with the website, but after experimenting with several methods, we were able to incorporate both the upload and live video options into our website. 

## Accomplishments that we're proud of
We are proud of being able to use MediaPipe to effectively map the location of various landmarks, like the nose or shoulders, using a video. Furthermore, we were able to use the location of these features to determine any form issues and how to correct them. We really hope that PET  can be used as a widespread tool to teach students about fitness!

## What we learned
We learned a lot about applications of pose-estimation models and using the MediaPipe library. We also learned more about web development such as taking in live video feed from a webcam.

## What's next for PET
We would like to expand upon our idea to provide real-time insights for more exercises and improve upon the teacher feedback functionality feature.
