# PET(Physical Education Tool)
## Inspiration
The four of us on the team experienced a full year of virtual learning during our sophomore of high school. Perhaps the class that was impacted the most was Physical Education or PE. It was extremely difficult for the teacher to monitor students’ form during various exercises, especially when there were 30 videos playing simultaneously. As an alternate method for feedback, our teachers requested video submissions and provided personal comments. We realized that this process was very tedious for teachers and made it difficult for students to improve their form for difficult exercises. As a result, we thought of PET (Physical Education Tool), a method to automate the feedback a teacher would normally provide to a student.

## What it does
PET is a software that can be used to help physical education teachers ensure their students are performing various exercises correctly through a virtual environment. The user uploads a video of them doing a squat, curl up, bicep curl, plank, and/or push up to the website, and PET provides real-time feedback of what, if anything, the user needs to improve about their form.

## How we built it
We developed the machine learning and computer vision algorithms using Google's MediaPipe and OpenCV.  The UI was created using Flask. We developed code using Google Colaboratory and Visual Studio Code before integrating individual pieces into the overall system using GitHub.

## Challenges we ran into
It was a little difficult to determine when the start and the end of one rep was.

## Accomplishments that we're proud of
We are proud of being able to use MediaPipe to effectively map the location of various landmarks, like the nose or shoulders, using a video. Furthermore, we were able to use the location of these features to determine any form issues and how to correct them. We really hope that PET  can be used as a widespread tool to teach students about fitness!

## What we learned
We learned a lot about applications of pose-estimation models and using the MediaPipe library. We also learned more about web development such as taking in live video feed from a webcam.

## What's next for PET
We would like to expand upon our idea to provide real-time insights for more exercises and improve upon the teacher feedback functionality feature.
