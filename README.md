# Project Goal

- This app aims to improve your **power bolwing (modern bowling)** technique by analyzing and suggesting helpful posture information.
  - Propose the standard of cranker-style 5 step approach to throw the bowling ball
  - Suggest pose analysis of users' video with mediapipe pose detection
 
# User Persona
  - I am a begigner, but want to be a cranker sytle bowler.
  - I am amateur experienced in stroker style for several years, but wanna change my style to cranker.
  - I don't wanna get injured performing wrong posture.

# MVP function
  - A user uploads his/her own bowling posture video. (Highly recommended to upload video taken right behind the bowler).
  - For each step, pose estimation and appreciation result is saved.
  - Analysis and evaluation of each step will be suggested with proper guide comments.

# Technical Requirements

1. segement each step by whether each foot is moving or stopped for each step (1,2,3,4,5) and get each frames of 1 to 5 step.
2. Implement analaysis logic for each power step
   - torso angle
     **3,4 step**
       torso angle should be tilted right shifting from 3 step to 4 step
     **4,5 step**
       torso angle in 4 step should be held until the bowl release of 5 step
   - foot position
      **2 step**
       right foot should be in line with the left foot
      **3 step**
       left foot should not overlap with the right foot in z-axis
      **4 step**
       right foot should overlap with the left foot in z-axis
       right foot should go forward as short as possible
      **5 step**
       First, we should hold all the weight of our body on right foot alone
       Second, left foot should be sliding while keeping the right foot weight entirely.
       Finally, the right foot weight should be shifted to left foot in a second.
3. Suggest correction text based on the criteria for each step
  
