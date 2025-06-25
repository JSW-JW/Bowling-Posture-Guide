# Project Goal

- This app aims to improve your **power bolwing (modern bowling)** technique by analyzing and suggesting helpful posture information.
  - Movement analysis of each step based on **5 power step** approach. 
  - Comparison between your posture and desirable posture.
 
# What is Power Bowling or Modern Bowling ?
It is tricky to describe exactly what is power bowling or modern bowling, but every bowler that has modern bowling style has some common things in their step and rythm.

## MVP function
    - A user uploads his/her own bowling posture video. (recommends video taken from right behind the bowler).
    - For each step, pose estimation and appreciation result is saved.
    - Analysis and evaluation of your performing each step will be suggested with proper guide comments.

## Requirements

1. segement each step based on velocity and acceleration of z-axis (the distance between camera and bowler)
2. analaysis logic for each power step (torso angle, foot position)
3. guide comments for each power step if the user's pose is right or wrong
  
