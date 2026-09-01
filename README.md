# AI-Based Predictive Road Safety and Driver Assistance System for Two-Wheelers

## 🚦 Overview

Two-wheelers are highly vulnerable to road accidents due to factors such as
rider drowsiness, distraction, excessive speed, sudden obstacles, road
hazards and loss of vehicle control.

This project proposes an **AI-Based Predictive Road Safety and Driver
Assistance System for Two-Wheelers** that combines **Artificial Intelligence,
Computer Vision, Embedded Systems and Vehicle Dynamics Analysis** to monitor
the rider, understand the surrounding environment, analyse vehicle behaviour
and provide timely safety warnings.

The system is being developed as a modular prototype, where individual
safety modules are developed and tested before being integrated into a
unified risk-assessment system.

---

# 🎯 Problem Statement

Two-wheeler accidents can occur because of multiple factors:

- Rider drowsiness or fatigue
- Driver distraction
- Excessive vehicle speed
- Sudden braking
- Unstable vehicle movement
- Loss of vehicle control
- Potholes and speed breakers
- Nearby vehicles or pedestrians
- Unexpected road obstacles
- Delayed reaction to hazards

Most basic safety systems focus on only one of these factors.

Our proposed system aims to combine **rider condition, environmental hazards
and vehicle dynamics** to estimate the overall risk and provide an
appropriate warning to the rider.

---

# 🎯 Objectives

The main objectives of the project are:

- Detect rider drowsiness in real time.
- Monitor eye closure and blinking behaviour.
- Detect yawning as an indicator of fatigue.
- Detect possible rider distraction using head orientation.
- Detect surrounding objects using computer vision.
- Detect road hazards such as potholes and speed breakers.
- Monitor vehicle speed and motion.
- Analyse acceleration and deceleration.
- Analyse gyroscope data and vehicle movement.
- Detect sudden braking and abnormal vehicle behaviour.
- Estimate vehicle stability and possible loss of control.
- Combine multiple safety parameters into a unified risk assessment.
- Provide real-time audible and/or haptic warnings.
- Develop a scalable prototype for intelligent two-wheeler safety.

---

# 🔑 Key Features

## 1. Rider Monitoring

The rider-monitoring module analyses the rider through a camera.

### Current capabilities

- Face detection
- Facial landmark detection
- Eye detection
- Eye Aspect Ratio (EAR)
- Blink detection
- Prolonged eye-closure detection
- Drowsiness detection
- Mouth opening analysis
- Yawning detection
- Head orientation monitoring
- Driver distraction detection

### Example
Camera
   ↓
Face Detection
   ↓
Facial Landmarks
   ↓
Eye + Mouth + Head Analysis
   ↓
Rider State
   ↓
Alert / Risk Analysis

                         TWO-WHEELER
                         SAFETY SYSTEM
                              │
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ↓                ↓                ↓
       RIDER MONITORING  ENVIRONMENT      VEHICLE DYNAMICS
             │           PERCEPTION             │
             │                │                 │
        ┌────┴────┐      ┌────┴────┐       ┌────┴─────┐
        │         │      │         │       │          │
      Eyes      Face    Objects   Hazards  Speed   Sensors
        │         │      │         │       │          │
      EAR      Head     Person    Pothole  Accel.   Gyroscope
      Blink    Position Car       Speed    Decel.   Lean Angle
      Yawn     Distraction Bike   Breaker  Braking  Motion
      Drowsy             etc.     etc.
        │         │      │         │       │          │
        └─────────┴──────┴─────────┴───────┴──────────┘
                              │
                              ↓
                    ┌────────────────────┐
                    │   PREDICTIVE       │
                    │   RISK ANALYSIS    │
                    └─────────┬──────────┘
                              │
                              ↓
                       RISK CLASSIFICATION
                              │
                   ┌──────────┼──────────┐
                   ↓          ↓          ↓
                  LOW       MEDIUM      HIGH
                              │
                              ↓
                    ┌──────────────────┐
                    │   ESP32 ALERT    │
                    │     SYSTEM       │
                    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    ↓                 ↓
                 🔊 BUZZER       📳 VIBRATION
