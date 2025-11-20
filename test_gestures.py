"""
Test script for gesture detection.

Run this to verify gesture detection is working correctly.
"""

import cv2
from gesture_detector import GestureDetector

def main():
    print("Starting gesture detection test...")
    print("Press 'q' to quit\n")
    
    # Initialize detector
    detector = GestureDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return 1
    
    print("Gesture detection active. Try these gestures:")
    print("  - Raise both hands above your head (hands_up)")
    print("  - Touch your temple with one hand (temple_tap)")
    print("  - Put your hand near your chin (thinking)")
    print("  - Raise your eyebrow like The Rock (eyebrow_raise)")
    print()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Detect gestures
        result = detector.detect(frame)
        
        # Draw hand landmarks (21 points per hand) and face indicators
        frame = detector.draw_landmarks(frame, result, draw_face=True)
        
        # Draw results text overlay
        y_offset = 30
        cv2.putText(
            frame,
            f"Gestures: {', '.join(result.gesture_tags)}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        
        y_offset += 30
        cv2.putText(
            frame,
            f"Confidence: {result.confidence:.2f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        
        y_offset += 30
        cv2.putText(
            frame,
            f"Hands detected: {len(result.hand_keypoints)}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        
        # Show hand labels
        for i, hand_kp in enumerate(result.hand_keypoints):
            y_offset += 25
            cv2.putText(
                frame,
                f"  {hand_kp.hand_label} hand: {len(hand_kp.landmarks)} landmarks",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
        
        # Show frame
        cv2.imshow("Gesture Detection Test", frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    
    print("\nTest complete!")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
