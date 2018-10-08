## Hand Gestures for Drone Control Using Deep Learning

- A Complete system to control the drone using hand gestures. The following video shows the result of this research:


[![Watch the video](https://github.com/SubhiH/hand_gesture_controller/blob/master/demo/demo_1.png)](https://www.youtube.com/watch?v=_vK-ca2MNX4)

- The system proposed in this research consists of three modules:
  1.  Hand Detector: SSD deep neural network detector is used to recognize and localize the hands. 
  The dataset was collected and labelled for this project and it contains ~3200 samples.It was acquired in outdoor and indoor enviromnents with one and two hands. The dataset is available in this repository [hand_dataset](https://github.com/SubhiH/hand_dataset)
  
  ![Samples](https://github.com/SubhiH/hand_gesture_controller/blob/master/demo/demo_2.png)
  
  2. Gestures Recognizer: OpenCV is used to build image processing algorithm to recognize the gestures. The user can contol the drone in a similair way he drives the car using virtual wheel. The arm and takeoff gesture is shown in the following figure.
  
    ![Samples](https://github.com/SubhiH/hand_gesture_controller/blob/master/demo/demo_3.png)
    
    Some samples of the movement gestures:
    
    ![Samples](https://github.com/SubhiH/hand_gesture_controller/blob/master/demo/demo_4.png)

  
  3.  Drone Controller: Ardupilot system is used in this module. A simple system is built depending on dronekit library and   MAVLink messages.



## Citing this work

If you want to cite this work, use the following.

Soubhi Hadri, Hand Gestures for Drone Control Using Deep Learning, GitHub repository, https://github.com/SubhiH/hand_gesture_controller
```bib
@misc{Soubhi2018,
  author = {Soubhi, Hadri},
  title = {Hand Gestures for Drone Control Using Deep Learning },
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/SubhiH/hand_gesture_controller}}
  }
```

## References

You can find the detailed list of resources at the end of the [thesis](https://github.com/SubhiH/hand_dataset).
