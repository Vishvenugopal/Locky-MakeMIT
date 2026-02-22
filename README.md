# 🤖 Locky --- The Phone-Locking Focus Robot

Locky is a physical commitment device designed to eliminate phone
distractions.\
Instead of relying on willpower, Locky removes temptation entirely ---
by locking your phone in a box and physically driving away to hide it.

Inspired by commitment devices and a little bit of TARS from
Interstellar, Locky turns focus into something physical, fun, and hard
to bypass.

------------------------------------------------------------------------

## 🚀 What It Does

1.  Start a focus session in the iOS app\
2.  Place your phone into the lock box mounted on Locky\
3.  Locky uses iPhone Pro LiDAR mapping to navigate\
4.  It drives to one of three preset hiding spots\
5.  It stays hidden until the timer ends\
6.  It returns (or unlocks) so you can retrieve your phone

No willpower required. No easy override.

------------------------------------------------------------------------

## 🧠 Why We Built It

Most productivity apps: - Can be deleted - Can be ignored - Can be
turned off in seconds

Locky creates physical friction.\
You're not fighting temptation every minute --- your phone is literally
gone.

------------------------------------------------------------------------

## 🏗 System Architecture

### 📱 iOS App (Swift + Xcode)

-   LiDAR mapping (iPhone Pro models)
-   Focus session control
-   Navigation trigger
-   Timer management

### 🤖 Mobile Robot

-   Raspberry Pi onboard controller
-   Controlled through Viam
-   Custom phone lock box mount
-   Upgraded LiPo battery
-   Rewired power supply

### 🌐 Viam Platform

-   Remote component control
-   Rapid robotics iteration
-   Reliable hardware abstraction

------------------------------------------------------------------------

## 🔧 Hardware Setup

Core Components: - Raspberry Pi - Mobile robot chassis - LiPo battery
upgrade - Custom phone holder + lock box - iPhone Pro (LiDAR-enabled)

------------------------------------------------------------------------

## ⚙️ Key Technical Features

-   LiDAR-based room scanning
-   Randomized hiding spot selection
-   Stable battery and power wiring
-   Viam-based robot control
-   Working iOS focus app
-   Secure lock box mechanism

------------------------------------------------------------------------

## 🧪 Challenges We Faced

-   Weak stock batteries → switched to LiPo
-   Improvised charging system
-   Unstable connectivity (personal hotspot)
-   iOS code signing issues
-   Hardware + mobile + cloud integration complexity

------------------------------------------------------------------------

## 🌟 Future Roadmap

-   Remote deployment (control from anywhere)
-   Telepresence mode (trusted friend can drive Locky with permission)
-   OpenAI voice integration for personality
-   Improved connectivity (local server / mesh network)
-   Enhanced security permissions

------------------------------------------------------------------------
