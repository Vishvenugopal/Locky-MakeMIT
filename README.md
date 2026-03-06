## About Locky
**Made at MakeMIT x Harvard 2026**

Locky is a robot that helps you fight your phone addiction. First, you give it your phone and it drives around your room and creates a mental map using the lidar sensor on your phone. Then, when you find yourself too glued to your phone, you can give your phone to locky and it chooses a smart and random spot in the room to hide. This spot is optimized to be far away and out of line-of-sight from you!

To build this we used the kindly provided Viam Rover! We attached a locking phone box to the rover. We used Viam software to control the rover through our scripts. We used XCode to develop the mobile app and python for the raspberry pi on board the rover.

Have fun playing hide and seek (and staying focused) with Locky!

Also thank you to Viam for providing us with the Rover for testing!

### Demo Video: https://www.youtube.com/watch?v=eusr-BnM674
<img width="2277" height="1361" alt="LockyVideoScreenshot" src="https://github.com/user-attachments/assets/fc8e2679-2eec-4835-97c6-72aacf488dc4" /> 

------------------------------------------------------------------------

## Usage

1.  Place your phone in Locky's custom phone holder, press start, and let Locky roam around and create a room scan
2.  Wait for Locky to return to you after scanning.
3.  When you want Locky to hide, put your phone inside, and Locky will go to a smart hiding spot outside of your view.
4.  Future plans: There could be a timer the user sets on the phone app, and once it ends Locky will return back to you.

------------------------------------------------------------------------

## Why We Built It

Most productivity apps: Can be deleted, ignore, or easily turned off.

Locky physically takes your phone away and hides!

------------------------------------------------------------------------

## System Architecture

### iOS App (Swift + Xcode)

-   Streaming LiDAR mapping (iPhone Pro models) to Robot
-   Start scan feature
-   Begin Hiding feature

### Mobile Robot (Viam Rover)

-   Raspberry Pi onboard controller
-   Controlled through Viam
-   Custom phone lock box mount
-   Upgraded LiPo battery
-   Rewired power supply

### Viam Platform

-   Remote component control
-   Rapid robotics iteration
-   Reliable hardware abstraction

------------------------------------------------------------------------

## Hardware Setup

Core Components: - Raspberry Pi - Mobile robot chassis - LiPo battery
upgrade - Custom phone holder + lock box - iPhone Pro (LiDAR-enabled)

------------------------------------------------------------------------

## Key Technical Features

-   LiDAR-based room scanning using iPhone Pro Models' LiDAR sensor
-   Randomized hiding spot selection
-   Stable battery and power wiring
-   Viam-based robot control
-   Working iOS focus app
-   Secure lock box mechanism

------------------------------------------------------------------------

## Challenges We Faced

-   Weak stock batteries → switched to LiPo
-   Improvised charging system
-   Unstable connectivity (personal hotspot)
-   iOS code signing issues
-   Hardware + mobile + cloud integration complexity

------------------------------------------------------------------------

## Future Roadmap

-   Remote deployment (control from anywhere)
-   Telepresence mode (trusted friend can drive Locky with permission)
-   OpenAI voice integration for personality
-   Improved connectivity (local server / mesh network)
-   Enhanced security permissions

------------------------------------------------------------------------
