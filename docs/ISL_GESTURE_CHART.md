# Indian Sign Language (ISL) 35-Class Real-Time Gesture Reference Guide

This document provides a practical, step-by-step visual reference for every gesture recognized by the **Indian Sign Language (ISL) Recognition System V2**.

---

## 🚀 Quick-Start: 6 Easiest Signs to Test First

If you want an instant 99% confidence test right now, try these **single-handed gestures** using your **Right hand**:

| Sign | Gesture Name | Finger Position | Confidence Target |
| :---: | :--- | :--- | :---: |
| **`1`** | Single Finger | Index finger pointing straight UP; fist closed. | $\ge 98\%$ |
| **`2`** | Two Fingers | Index + Middle fingers pointing straight UP in a narrow "V". | $\ge 98\%$ |
| **`5`** | Open Palm | All 5 fingers spread open facing the camera. | $\ge 95\%$ |
| **`L`** | L-Shape | Thumb extended horizontally, Index finger pointing UP ($90^\circ$ angle). | $\ge 99\%$ |
| **`C`** | C-Curve | Hand curved like holding a cup, thumb and fingers curved towards each other. | $\ge 96\%$ |
| **`V`** | Victory / Peace | Index and Middle fingers spread in a wide "V", thumb over ring/pinky. | $\ge 97\%$ |

---

## 🔢 Category 1: Digits 1 to 9 (Single Hand — Dominant / Right)

In Indian Sign Language, counting digits 1–9 is performed using a single hand held upright in front of the chest, palm facing forward toward the webcam:

| Digit | Hand Position | Visual Description |
| :---: | :--- | :--- |
| **`1`** | Right Hand | **Index finger** extended straight UP; thumb and other 3 fingers curled into a fist. |
| **`2`** | Right Hand | **Index and Middle fingers** extended UP; ring, pinky, and thumb folded in. |
| **`3`** | Right Hand | **Index, Middle, and Ring fingers** extended UP; pinky folded, thumb over pinky. |
| **`4`** | Right Hand | **Four fingers** (Index, Middle, Ring, Pinky) extended UP; thumb tucked across palm. |
| **`5`** | Right Hand | **All 5 fingers** fully extended and spread outward facing the camera. |
| **`6`** | Right Hand | Make a fist with only the **Thumb** extended outward (or thumb + pinky). |
| **`7`** | Right Hand | **Thumb and Index finger** touching or extended forward; remaining fingers curled. |
| **`8`** | Right Hand | **Index, Middle fingers + Thumb** extended forward together. |
| **`9`** | Right Hand | **Index finger curved** like a hook (or closed fist with little finger and ring extended). |
| **`0`** | *Unsupported* | **Absent from training dataset** — deliberately documented as unsupported. |

---

## 🔤 Category 2: Single-Handed Alphabets

These letters only require **one hand** (dominant hand) in front of the camera:

| Letter | Hand Shape | Detailed Finger Configuration |
| :---: | :--- | :--- |
| **`C`** | Cup Curve | Curve all four fingers and thumb into a clear "**C**" arc facing sideways. |
| **`I`** | Little Finger | Closed fist with only the **Pinky (little finger)** pointing straight UP. |
| **`L`** | Right Angle | **Thumb** pointing sideways, **Index** pointing straight UP ($90^\circ$ "L" shape). |
| **`O`** | Circle Ring | All four fingertips curved to touch the tip of the thumb, making an "**O**". |
| **`V`** | Peace Sign | **Index and Middle** fingers extended upward and spread into a "**V**". |
| **`Y`** | Phone / Hang Loose | **Thumb and Pinky** extended outward; middle 3 fingers folded into fist. |
| **`H`** | Horizontal Cut | Hand flat, fingers extended horizontally across chest. |
| **`P`** | Downward Loop | Index finger pointed downward with thumb touching first knuckle. |
| **`U`** | Two Fingers Closed | Index and Middle fingers extended straight UP touching side-by-side (unlike V). |
| **`Z`** | Zig-Zag Pose | Index finger extended pointing diagonally forward/up. |

---

## 👐 Category 3: Dual-Handed Alphabets (Two Hands — Traditional ISL)

Indian Sign Language (derived from BANZSL) represents many letters using **two hands**, where the **Left Hand acts as the passive base/paper** and the **Right Hand acts as the active pen**:

### The ISL Vowel Rule (A, E, I, O, U)
In ISL, the 5 vowels are formed by touching the fingertips of the passive Left Hand with the Right Index Finger:
* **`A`**: Right index finger touches the tip of the **Left Thumb**.
* **`E`**: Right index finger touches the tip of the **Left Index Finger**.
* **`I`**: Right index finger touches the tip of the **Left Middle Finger** (or single pinky UP).
* **`O`**: Right index finger touches the tip of the **Left Ring Finger** (or single-hand "O" circle).
* **`U`**: Right index finger touches the tip of the **Left Pinky Finger**.

### Two-Hand Consonants
| Letter | Formation | Step-by-Step Instructions |
| :---: | :--- | :--- |
| **`A`** | Vowel Touch | Left hand open palm facing you; **Right index touches Left Thumb tip**. |
| **`B`** | Double Loop | Both hands form circles (touching thumbs and index fingers) held side-by-side. |
| **`D`** | Spine + Arc | Left index held vertically; Right hand forms a "**C**" touching top and base of Left index. |
| **`E`** | Vowel Touch | Left hand open palm facing you; **Right index touches Left Index tip**. |
| **`G`** | Double Fist | Both hands in closed fists placed one on top of the other. |
| **`K`** | Angle Touch | Left index held vertical; Right index bent at knuckle touching the side of Left index. |
| **`M`** | Three Legs | Left hand flat palm up; Right hand places **3 fingers** (Index, Middle, Ring) down on Left palm. |
| **`N`** | Two Legs | Left hand flat palm up; Right hand places **2 fingers** (Index, Middle) down on Left palm. |
| **`S`** | Interlocked Pinkies | Left and Right pinkies hooked together, or Right fist resting on Left fist. |
| **`T`** | T-Intersection | Left hand held flat vertically; **Right index finger touches the edge** forming a "**T**". |
| **`W`** | Interlocked Fingers | Both hands with fingers interlocked and spread upward, forming "**W**". |
| **`X`** | Cross Bones | Both index fingers crossed over each other at $90^\circ$ forming an "**X**". |

---

## 🎯 Best Real-World Operating Tips

1. **Distance:** Sit **0.8m to 1.2m** away from the camera so your hands and chest are clearly in frame.
2. **Lighting:** Avoid sitting with a bright window behind you (backlighting). Front lighting gives MediaPipe the clearest hand contours.
3. **The 0.8s Commit Hold:**
   * When you form a sign, the UI shows **`HOLDING`** (yellow).
   * Keep your hand still for **0.8 seconds** (~12 frames) until the progress bar fills to 100%.
   * The character commits to the sentence buffer and turns **`COMMITTED`** (green), then **`LOCKED`** (purple).
4. **Typing Words:**
   * To type `CAT`: Hold `C` $\to$ relax hand $\to$ hold `A` $\to$ relax hand $\to$ hold `T`.
   * Click **`[Space]`** to separate words!
