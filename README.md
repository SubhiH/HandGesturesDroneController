## Hand Gestures for Drone Control Using Deep Learning 🙋:fist:  :hand:  :point_up:  :raised_hands:


[![Apache](https://img.shields.io/badge/License-Apache--2.0-red.svg)](https://opensource.org/licenses/Apache-2.0)
[![Flight Stack](https://img.shields.io/badge/Flight%20Stack-Ardupilot-blue.svg)](http://ardupilot.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-3.4-red.svg)](https://pypi.org/project/opencv-python/)
[![Python](https://img.shields.io/badge/Python-2.7-red.svg)](https://docs.python.org/release/2.7.15/)
[![Tensorflow](https://img.shields.io/badge/TensorFlow-1.11-red.svg)](https://www.tensorflow.org/)
[![SSD](https://img.shields.io/badge/Detector-SSD-yellowgreen.svg)](https://arxiv.org/abs/1512.02325)
[![Docker Build](https://img.shields.io/docker/automated/jrottenberg/ffmpeg.svg)](https://github.com/SubhiH/hand_gesture_controller/blob/master/Dockerfile)

- A Complete system to control the drone using hand gestures. The following video shows the result of this research:


[![Watch the video](https://github.com/SubhiH/hand_gesture_controller/blob/master/demo/demo_1.png)](https://www.youtube.com/watch?v=_vK-ca2MNX4)

---

- The proposed system consists of three modules:

---

  1.  **Hand Detector**: *SSD* deep neural network detector is used to recognize and localize the hands. 
  The dataset was collected and labelled for this project. It contains ~3200 samples acquired in outdoor and indoor enviromnents with one and two hands. The dataset is available in this repository [**HandsDataset**](https://github.com/SubhiH/hand_dataset)
  
  ![Samples](https://github.com/SubhiH/hand_gesture_controller/blob/master/demo/demo_2.png)
  

---


  2. **Gestures Recognizer**: Image processing algorithm is developed to recognize the gestures. The user can contol the drone in a similar way he drives the car using virtual wheel. The arm and takeoff gesture is shown in the following figure.
  
   ![Samples](https://github.com/SubhiH/hand_gesture_controller/blob/master/demo/demo_3.png)
    
    Some samples of the movement gestures:
    
   ![Samples](https://github.com/SubhiH/hand_gesture_controller/blob/master/demo/demo_4.png)
   
---


  3.  **Drone Controller**: Ardupilot system is used in this module. A simple system is built using dronekit library and MAVLink messages.


---

## Notes:

1.  Research paper of this work is under review.
2.  Thesis of this work will be submitted to University of Oklahoma at the end of November and will be available online.
3.  The system is tested with CPU and works in real-time.
4.  Everyone is welcome to contribute :candy: :doughnut: :ice_cream: .


### TO-DO :

- [ ] Star the repository :star: :wink:.
- [ ] Installation instructions.
- [x] Upload dataset.
- [x] Upload trained model.
- [x] Autpilot repo.
- [x] GUI.
- [ ] Thesis link :blue_book: .
- [ ] Paper link :page_facing_up:.


## Citing this work

If you want to cite this work, use the following:

Soubhi Hadri, Hand Gestures for Drone Control Using Deep Learning, GitHub repository, https://github.com/SubhiH/HandGesturesDroneController
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

## Thanks: :pray:

Special thanks to:

- [Ardupilot Team and Community](http://ardupilot.org/about/team).
- [Harrison](https://github.com/Sentdex).


## References

You can find the detailed list of resources at the end of the [thesis](https://github.com/SubhiH/hand_dataset).


## License

HandGesturesDroneController is released under the Apache license. See [LICENSE](https://github.com/SubhiH/hand_gesture_controller/blob/master/LICENSE) for more information.
