# Project Goal

- This app aims to improve your **power bolwing (modern bowling)** technique by analyzing and suggesting helpful posture information.
  - Propose the standard of cranker-style 5 step approach to throw the bowling ball
  - Suggest pose analysis of users' video with mediapipe pose detection
 
# What is Power Bowling or Modern Bowling ?
It is tricky to describe exactly what is power bowling or modern bowling, but every bowler that has modern bowling style has some common things in their **step** and **rythm**.

## Step

## Rythm

# MVP function
    - A user uploads his/her own bowling posture video. (recommends video taken from right behind the bowler).
    - For each step, pose estimation and appreciation result is saved.
    - Analysis and evaluation of your performing each step will be suggested with proper guide comments.

# Technical Requirements

1. segement each step by whether each foot is moving or stopped for each step (1,2,3,4,5) and get each frames of 1 to 5 step.
2. analaysis logic for each power step
    - torso angle
      [3,4 step](torso angle should be tilted right shifting from 3 step to 4 step)
      torso angle in 4 step should be held until the bowl release of 5 step
    - foot position
      2 step: right foot should be in line with the left foot
      3 step:
4. guide comments for each power step if the user's pose is right or wrong
  
