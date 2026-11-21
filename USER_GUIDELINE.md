# 🤟 Indian Sign Language (ISL) Real-Time Operating & Gesture Guide
### Complete Reference for Digits (1–9) & Alphabets (A–Z)

**Project:** AI Indian Sign Language Recognition System V2  
**Target Audience:** Users, Evaluators, and Live Demo Presenters  
**Language:** English with Easy Hinglish Context  

---

## 📌 Table of Contents
1. [⚡ Quick-Start Rules for 99% Accuracy](#-quick-start-rules-for-99-accuracy)
2. [🖥️ UI Modes: Digits vs Alphabets vs All](#-ui-modes-digits-vs-alphabets-vs-all)
3. [🎯 Camera Setup & Lighting Essentials](#-camera-setup--lighting-essentials)
4. [⏱️ The Gesture Lifecycle: How Character Commit Works](#-the-gesture-lifecycle-how-character-commit-works)
5. [🔢 Complete Digits Guide (1 to 9)](#-complete-digits-guide-1-to-9)
6. [🔤 Complete Alphabets Guide (A to Z)](#-complete-alphabets-guide-a-to-z)
   - [Part A: Dual-Handed Alphabets (2 Hands Required!)](#part-a-dual-handed-alphabets-2-hands-required)
   - [Part B: Single-Handed Alphabets (1 Hand)](#part-b-single-handed-alphabets-1-hand)
7. [⚠️ Common Mistakes & Instant Fixes](#-common-mistakes--instant-fixes)

---

## ⚡ Quick-Start Rules for 99% Accuracy

If you follow these **3 fundamental rules**, the system will recognize your signs with **95% to 99% accuracy**:

1. **Know 1-Hand vs 2-Hand Signs:**  
   Indian Sign Language (ISL) is **NOT** American Sign Language (ASL).  
   - In ISL, letters like **A, B, D, E, G, K, M, N, T, X, Y** require **BOTH HANDS** in front of the camera. If you show only 1 hand for 'A', the model cannot detect it!
   - Single-hand signs (like **C, H, I, L, O, P, U, V, W**) only need your **dominant hand**.
2. **Select the Right Vocabulary Mode:**  
   - When practicing or typing **Numbers**, select `[Digits (1–9)]` mode on the top-right of the screen.
   - When practicing or typing **Letters**, select `[Alphabets (A–Z)]` mode.
   - When typing a full mixed sentence, use `[All 35 Signs]`.
3. **The "Hold & Relax" Rhythm:**  
   - Hold your sign still for **0.6 to 0.8 seconds** until the Commit Hold progress bar fills and flashes **COMMITTED** (green).
   - **Relax or drop your hand for half a second** before forming the next sign. This gives the system time to reset its debounce lock and prevents duplicate letters or transitional false triggers.

---

## 🖥️ UI Modes: Digits vs Alphabets vs All

The web app includes a **Vocabulary Switcher** in the top navigation bar:

| Mode | What It Does | When to Use |
| :--- | :--- | :--- |
| **`[Digits (1–9)]`** | Masks out all 26 alphabets. Evaluates only digits 1 to 9. | Use when testing numbers, PINs, or counting. Eliminates all false letters (like 'V' or 'Z'). |
| **`[Alphabets (A–Z)]`** | Masks out all 9 digits. Evaluates only letters A to Z. | Use when spelling names, words, or testing letters. Prevents digits (like '1' or '2') from stealing letters. |
| **`[All 35 Signs]`** | Evaluates all 35 classes simultaneously with intelligent conflict resolution. | Use for natural mixed conversation and end-to-end demonstrations. |

---

## 🎯 Camera Setup & Lighting Essentials

1. **Distance:** Sit between **0.8 meter to 1.2 meters** (about 2.5 to 4 feet) from your webcam.
   - Your **chest, shoulders, and both hands** must be comfortably visible within the camera view.
2. **Lighting:**
   - **Good front lighting** is essential. Light should fall directly on your face and hands.
   - **Avoid backlighting:** Never sit with a bright window or bright lamp behind you. Backlight turns hands into dark silhouettes and degrades MediaPipe 3D joint tracking.
3. **Background:** A relatively neutral, plain background (wall, curtain, or uncluttered room) provides the best skeleton tracking.

---

## ⏱️ The Gesture Lifecycle: How Character Commit Works

The system uses a **4-Stage Finite State Machine (FSM)** with temporal smoothing to ensure you never get accidental triggers:

```
[ IDLE ] ──(Hand Detected)──> [ HOLDING ] ──(Hold ~0.8s)──> [ COMMITTED ] ──> [ LOCKED ]
   ▲                                                                              │
   └───────────────(Relax hand / Lower hand for ~0.4s)───────────────────────────┘
```

1. **`IDLE` (Grey/Cyan):** Camera is scanning; ready for your hand.
2. **`HOLDING` (Amber/Yellow):** The system sees your sign. The **Commit Hold** progress bar fills from 0% to 100%. Keep your hand steady.
3. **`COMMITTED` (Green):** When the bar reaches 100%, an audio tick plays, and the letter is committed to your sentence buffer.
4. **`LOCKED` (Purple):** Prevents the same letter from typing 10 times while your hand stays up.
5. **Next Letter Reset:** Simply relax or lower your hand for **0.4 seconds** (idle debounce threshold). The state resets to `IDLE`, ready for your next character!

---

## 🔢 Complete Digits Guide (1 to 9)

> **Important Note:** In Indian Sign Language counting, digits 1–5 use extended fingers, while digits 6–9 use specific thumb/pinky configurations. Digit `0` is absent from the dataset and unsupported.

Use your **Dominant Hand (Right Hand recommended)**, palm facing the camera:

| Digit | Hand Position | Visual Finger Configuration |
| :---: | :--- | :--- |
| **`1`** | Single Index | **Index finger pointing straight UP**; thumb and remaining 3 fingers closed into a fist. |
| **`2`** | Two Fingers | **Index + Middle finger pointing UP**; thumb folded over ring and pinky. |
| **`3`** | Three Fingers | **Index + Middle + Ring finger pointing UP**; pinky folded, thumb over pinky. |
| **`4`** | Four Fingers | **All 4 fingers (Index, Middle, Ring, Pinky) UP**; thumb tucked tightly across palm. |
| **`5`** | Open Palm | **All 5 fingers spread wide open** facing the camera (open hand). |
| **`6`** | Pinky Only | **Little finger (Pinky) pointing straight UP**; all other 4 fingers and thumb folded into a fist. *(Standard Asian fingerspelling counting)*. |
| **`7`** | Thumb + Index | **Thumb and Index finger extended outward** forming an 'L' shape (like pointing a gun upward); remaining 3 fingers folded. |
| **`8`** | Thumb + Index + Middle | **Thumb, Index, and Middle fingers extended** outward; ring and pinky folded in. |
| **`9`** | Thumb Only | **Thumb pointing straight UP / outward** (thumbs-up gesture); all 4 fingers folded into a fist. |

> **💡 Pro-Tip for Digits:** When you finish signing `5` and want to lower your hand, relax your hand inward rather than tucking fingers one-by-one. If you tuck 4 fingers while leaving the thumb sticking out, the system will briefly see `9`!

---

## 🔤 Complete Alphabets Guide (A to Z)

### Part A: Dual-Handed Alphabets (2 Hands Required!)
Indian Sign Language uses a **Two-Handed System** (derived from BANZSL) where:
- The **Left Hand** acts as the passive base/paper.
- The **Right Hand** acts as the active pen/pointer.

| Letter | Sign Name | How to Perform (Both Hands Visible) |
| :---: | :--- | :--- |
| **`A`** | Vowel: Thumb Touch | **Left hand open palm** facing you. Touch the tip of your **Left Thumb** with your **Right Index finger**. |
| **`B`** | Double Loop | Both hands form small circles (thumb and index tips touching). Place the two circles side-by-side (looks like eyeglasses or the number '8'). |
| **`D`** | Lowercase 'd' | **Left index finger pointing straight UP**. With your **Right hand**, make a semi-circle (arch) touching the top and bottom of the Left index to form a letter 'd'. |
| **`E`** | Vowel: Index Touch | **Left hand open palm** facing you. Touch the tip of your **Left Index finger** with your **Right Index finger**. |
| **`G`** | Double Fist | Both hands form closed fists. Place your **Right fist on top of your Left fist**. |
| **`K`** | K-Angle | **Left index finger pointing straight UP**. Bend your **Right index finger** and touch the middle of the Left index at a $45^\circ$ angle (forming the letter 'K'). |
| **`M`** | Three Legs | **Left hand flat palm facing UP/forward**. Place **3 fingers of your Right hand** (Index, Middle, Ring) down onto the Left palm (like the 3 legs of 'm'). |
| **`N`** | Two Legs | **Left hand flat palm facing UP/forward**. Place **2 fingers of your Right hand** (Index, Middle) down onto the Left palm (like the 2 legs of 'n'). |
| **`T`** | T-Intersection | **Left hand flat vertically**. Touch the edge of your Left hand with your **Right index finger** to form a perpendicular 'T'. |
| **`X`** | Cross Bones | **Cross both index fingers** over each other at a $90^\circ$ angle to make an 'X'. |
| **`Y`** | Palm Web | **Left hand open palm**. Place your **Right index finger into the web** between your Left thumb and Left index finger. |

---

### Part B: Single-Handed Alphabets (1 Hand)
These signs require only **one hand** (Dominant / Right hand):

| Letter | Sign Name | How to Perform (1 Hand) |
| :---: | :--- | :--- |
| **`C`** | Cup Curve | Curve all 4 fingers and thumb into a clear sideways "**C**" arc (as if holding a soda can). |
| **`H`** | Horizontal Cut | Hand held flat, **Index and Middle fingers extended horizontally** across your chest, thumb folded under. |
| **`I`** | Little Finger | Closed fist with only the **Pinky (little finger)** pointing straight UP. |
| **`J`** | Pinky Hook | Little finger extended, drawing a small downwards curve/hook in the air. |
| **`L`** | L-Shape | **Thumb extended horizontally, Index pointing straight UP** ($90^\circ$ standard 'L'). |
| **`O`** | Circle Ring | Curve all fingers to meet the thumb tip, forming a closed circular "**O**". |
| **`P`** | Downward Loop | Index finger pointed downwards, with thumb and middle finger forming a loop. |
| **`Q`** | Downward Hook | Hand pointing downwards forming an open pincher/hook shape. |
| **`R`** | Crossed Fingers | **Index and Middle fingers crossed over each other** (the classic "good luck" gesture). |
| **`S`** | Closed Fist | Form a **tight, solid closed fist** facing the camera with thumb wrapped over knuckles. |
| **`U`** | Two Fingers Touching | **Index and Middle fingers extended straight UP touching together side-by-side** (unlike 'V' which is spread open). |
| **`V`** | Victory / Peace | **Index and Middle fingers extended and spread apart** in a wide "V". |
| **`W`** | Three Fingers Spread | **Index, Middle, and Ring fingers extended and spread apart** into a "W". |
| **`Z`** | Zig-Zag | Index finger pointing forward/diagonally. |

---

## ⚠️ Common Mistakes & Instant Fixes

### 1. "I am signing 'A' or 'B', but it detects nothing or something random!"
- **Cause:** You are probably using only 1 hand (like American Sign Language).
- **Fix:** In Indian Sign Language, **A and B are TWO-HANDED signs**. For `A`, touch your left thumb tip with your right index finger. Both hands must be visible in the camera frame!

### 2. "When I sign '2', it detects 'V'!"
- **Cause:** In physical anatomy, '2' (two fingers up) and 'V' (peace sign) have very similar landmarks.
- **Fix:** 
  - If you are practicing numbers, switch the UI Vocabulary Mode to **`[Digits (1–9)]`**. In Digits mode, 'V' is completely disabled and '2' will trigger 100% of the time.
  - In `[All 35 Signs]` mode, keep your index and middle fingers closer together and parallel for `2`, and spread them wide apart for `V`.

### 3. "Letters with repeated characters (e.g. 333 instead of 3)!"
- **Cause:** Dropping or twitching your hand before the FSM lock settles.
- **Fix:** Our v2.1 update includes **Idle Debounce**. After a sign commits, simply lower or relax your hand for 0.4s before raising it again.

### 4. "Digits 7, 8, or 9 appear when I relax my hand after 5!"
- **Cause:** In ISL, `9` is Thumb-only and `7` is Thumb+Index. When dropping a hand from `5`, leaving the thumb extended triggers `9` or `7`.
- **Fix:** Relax all 5 fingers together or drop your hand out of frame quickly.

---

## 🚀 Live Demo Practice Routine

To demonstrate the system with 100% confidence to teachers, interviewers, or peers:

1. **Step 1 (Digits Demo):**
   - Click **`[Digits (1–9)]`**.
   - Sign `1` $\to$ hold $\to$ commit $\to$ relax hand.
   - Sign `2` $\to$ hold $\to$ commit $\to$ relax hand.
   - Sign `3` $\to$ hold $\to$ commit $\to$ relax hand.
   - Sign `4` $\to$ hold $\to$ commit $\to$ relax hand.
   - Sign `5` $\to$ hold $\to$ commit.
   - Result: `12345` appears cleanly in the sentence buffer!

2. **Step 2 (Alphabets Demo):**
   - Click **`[Alphabets (A–Z)]`**.
   - Clear buffer.
   - Sign **`C`** (1 hand curve) $\to$ commits `C`.
   - Sign **`A`** (Right index on Left thumb) $\to$ commits `A`.
   - Sign **`T`** (Right index on Left hand edge) $\to$ commits `T`.
   - Click **`[Space]`**.
   - Sign **`I`** (Pinky up) $\to$ commits `I`.
   - Sign **`S`** (Closed fist) $\to$ commits `S`.
   - Result: `CAT IS` cleanly typed on screen!

---

*Indian Sign Language Recognition System V2 — Built with Python, MediaPipe & TensorFlow.*
