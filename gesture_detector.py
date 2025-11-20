"""
Gesture detection module using MediaPipe Holistic.

Detects hand, body, and face gestures with full landmark tracking.
Requires Python 3.8-3.12 for MediaPipe support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HandKeypoints:
    """Full hand landmarks from MediaPipe (21 points per hand)."""
    landmarks: list[tuple[float, float, float]]  # (x, y, z) normalized coordinates
    hand_label: str  # 'Left' or 'Right'
    world_landmarks: Optional[list[tuple[float, float, float]]] = None  # 3D world coordinates


@dataclass
class GestureResult:
    """Container for detected gestures and their confidence scores."""
    
    gesture_tags: list[str]
    confidence: float
    landmarks_detected: dict[str, bool]  # which landmark sets were found
    hand_keypoints: list[HandKeypoints]  # Full hand landmarks for each detected hand
    face_landmarks: Optional[any] = None  # MediaPipe face landmarks for visualization
    raw_data: Optional[dict] = None  # for debugging


class GestureDetector:
    """
    Detects gestures using MediaPipe Holistic (hands + pose + face).
    
    Supported gestures:
    - hands_up: Both hands raised above shoulders
    - temple_tap: Index finger near temple
    - thinking: Hand near chin/mouth
    - neutral: No specific gesture detected
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_face_mesh: bool = True,
    ):
        """
        Initialize the gesture detector.
        
        Args:
            min_detection_confidence: Minimum confidence for initial detection
            min_tracking_confidence: Minimum confidence for tracking
            enable_face_mesh: Whether to enable detailed face mesh (468 landmarks)
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            enable_segmentation=False,  # we don't need background segmentation
            refine_face_landmarks=enable_face_mesh,
        )
        
        # Landmark indices (MediaPipe Pose)
        self.POSE_LANDMARKS = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_index': 19,
            'right_index': 20,
            'mouth_left': 9,
            'mouth_right': 10,
        }
        
        # Hand landmarks (MediaPipe Hands) - 21 points
        self.HAND_LANDMARKS = {
            'wrist': 0,
            'thumb_cmc': 1,
            'thumb_mcp': 2,
            'thumb_ip': 3,
            'thumb_tip': 4,
            'index_mcp': 5,
            'index_pip': 6,
            'index_dip': 7,
            'index_tip': 8,
            'middle_mcp': 9,
            'middle_pip': 10,
            'middle_dip': 11,
            'middle_tip': 12,
            'ring_mcp': 13,
            'ring_pip': 14,
            'ring_dip': 15,
            'ring_tip': 16,
            'pinky_mcp': 17,
            'pinky_pip': 18,
            'pinky_dip': 19,
            'pinky_tip': 20,
        }
        
        logger.info("GestureDetector initialized with MediaPipe Holistic")
    
    def detect(self, frame: np.ndarray) -> GestureResult:
        """
        Detect gestures in a single frame.
        
        Args:
            frame: BGR image from OpenCV (H, W, 3)
            
        Returns:
            GestureResult with detected gesture tags and confidence
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.holistic.process(rgb_frame)
        
        # Track which landmark sets were detected
        landmarks_detected = {
            'pose': results.pose_landmarks is not None,
            'face': results.face_landmarks is not None,
            'left_hand': results.left_hand_landmarks is not None,
            'right_hand': results.right_hand_landmarks is not None,
        }
        
        # Extract hand keypoints
        hand_keypoints_list = []
        if results.left_hand_landmarks:
            hand_keypoints_list.append(self._extract_hand_keypoints(
                results.left_hand_landmarks,
                'Left'
            ))
        if results.right_hand_landmarks:
            hand_keypoints_list.append(self._extract_hand_keypoints(
                results.right_hand_landmarks,
                'Right'
            ))
        
        # Extract gestures
        gestures = []
        confidence = 0.0
        
        if results.pose_landmarks:
            pose_gestures, pose_conf = self._detect_pose_gestures(
                results.pose_landmarks,
                results.left_hand_landmarks,
                results.right_hand_landmarks,
                results.face_landmarks,  # Pass face landmarks for eyebrow detection
            )
            gestures.extend(pose_gestures)
            confidence = max(confidence, pose_conf)
        
        # Default to neutral if no specific gesture detected
        if not gestures:
            gestures = ['neutral']
            confidence = 0.3
        
        return GestureResult(
            gesture_tags=gestures,
            confidence=confidence,
            landmarks_detected=landmarks_detected,
            hand_keypoints=hand_keypoints_list,
            face_landmarks=results.face_landmarks,  # Store for visualization
            raw_data={
                'pose_present': results.pose_landmarks is not None,
                'hands_present': (
                    results.left_hand_landmarks is not None
                    or results.right_hand_landmarks is not None
                ),
            },
        )
    
    def _extract_hand_keypoints(
        self,
        hand_landmarks,
        hand_label: str,
    ) -> HandKeypoints:
        """Extract all 21 hand landmarks."""
        landmarks = [
            (lm.x, lm.y, lm.z)
            for lm in hand_landmarks.landmark
        ]
        
        return HandKeypoints(
            landmarks=landmarks,
            hand_label=hand_label,
            world_landmarks=None,  # Not available in Holistic
        )
    
    def _detect_pose_gestures(
        self,
        pose_landmarks,
        left_hand_landmarks,
        right_hand_landmarks,
        face_landmarks=None,
    ) -> tuple[list[str], float]:
        """
        Detect gestures from pose, hand, and face landmarks.
        
        Returns:
            (gesture_tags, confidence)
        """
        gestures = []
        max_confidence = 0.0
        
        # Extract key landmarks
        landmarks = pose_landmarks.landmark
        
        # Get normalized coordinates (0-1 range)
        left_wrist = landmarks[self.POSE_LANDMARKS['left_wrist']]
        right_wrist = landmarks[self.POSE_LANDMARKS['right_wrist']]
        left_shoulder = landmarks[self.POSE_LANDMARKS['left_shoulder']]
        right_shoulder = landmarks[self.POSE_LANDMARKS['right_shoulder']]
        nose = landmarks[self.POSE_LANDMARKS['nose']]
        
        # Check for eyebrow raise using face landmarks
        if face_landmarks:
            eyebrow_raised, eyebrow_conf = self._detect_eyebrow_raise(face_landmarks)
            if eyebrow_raised:
                gestures.append('eyebrow_raise')
                max_confidence = max(max_confidence, eyebrow_conf)
        
        # Check for "hands_up" - both wrists above shoulders
        if (
            left_wrist.y < left_shoulder.y
            and right_wrist.y < right_shoulder.y
            and left_wrist.visibility > 0.5
            and right_wrist.visibility > 0.5
        ):
            gestures.append('hands_up')
            max_confidence = max(
                max_confidence,
                min(left_wrist.visibility, right_wrist.visibility),
            )
        
        # Check for "temple_tap" - index finger near temple
        # Temple is roughly at eye level, slightly to the side of the nose
        temple_y = nose.y - 0.05  # slightly above nose
        
        for wrist, shoulder, hand_landmarks, side in [
            (left_wrist, left_shoulder, left_hand_landmarks, 'left'),
            (right_wrist, right_shoulder, right_hand_landmarks, 'right'),
        ]:
            if hand_landmarks and wrist.visibility > 0.5:
                # Get index finger tip
                index_tip = hand_landmarks.landmark[self.HAND_LANDMARKS['index_tip']]
                
                # Check if index finger is near temple (at head level, to the side)
                near_temple = (
                    abs(index_tip.y - temple_y) < 0.1
                    and index_tip.y < shoulder.y  # above shoulder
                )
                
                if near_temple:
                    gestures.append('temple_tap')
                    max_confidence = max(max_confidence, wrist.visibility)
                    break
        
        # Check for "thinking" - index finger near chin/mouth
        # Use face landmarks for more accurate chin position if available
        if face_landmarks:
            # Chin landmark is around index 152
            chin = face_landmarks.landmark[152]
            mouth_center = face_landmarks.landmark[13]  # Lower lip center
            chin_y = (chin.y + mouth_center.y) / 2
            chin_x = (chin.x + mouth_center.x) / 2
        else:
            # Fallback to pose-based estimation
            chin_y = nose.y + 0.08  # below nose
            chin_x = nose.x
        
        for wrist, hand_landmarks in [
            (left_wrist, left_hand_landmarks),
            (right_wrist, right_hand_landmarks),
        ]:
            if hand_landmarks and wrist.visibility > 0.5:
                # Use index fingertip for more precise detection
                index_tip = hand_landmarks.landmark[self.HAND_LANDMARKS['index_tip']]
                
                # Check if index finger is near chin/mouth area
                near_chin = (
                    abs(index_tip.y - chin_y) < 0.08
                    and abs(index_tip.x - chin_x) < 0.12
                )
                
                if near_chin:
                    gestures.append('thinking')
                    max_confidence = max(max_confidence, wrist.visibility)
                    break
        
        return gestures, max_confidence
    
    def _detect_eyebrow_raise(self, face_landmarks) -> tuple[bool, float]:
        """
        Detect eyebrow raise (like The Rock's signature move).
        
        Uses face landmarks to measure the distance between eyebrow and eye.
        When eyebrows are raised, this distance increases.
        
        Returns:
            (is_raised, confidence)
        """
        # Key face landmark indices for eyebrow detection
        # Right eyebrow (from camera view, user's right)
        RIGHT_EYEBROW_TOP = 70  # Top of right eyebrow
        RIGHT_EYE_TOP = 159     # Top of right eye
        
        # Left eyebrow (from camera view, user's left)
        LEFT_EYEBROW_TOP = 300  # Top of left eyebrow
        LEFT_EYE_TOP = 386      # Top of left eye
        
        # Get landmarks
        lms = face_landmarks.landmark
        
        # Calculate vertical distance between eyebrow and eye (both sides)
        right_eyebrow_y = lms[RIGHT_EYEBROW_TOP].y
        right_eye_y = lms[RIGHT_EYE_TOP].y
        right_distance = abs(right_eyebrow_y - right_eye_y)
        
        left_eyebrow_y = lms[LEFT_EYEBROW_TOP].y
        left_eye_y = lms[LEFT_EYE_TOP].y
        left_distance = abs(left_eyebrow_y - left_eye_y)
        
        # Average distance
        avg_distance = (right_distance + left_distance) / 2
        
        # Threshold for eyebrow raise (calibrated empirically)
        # Normal: ~0.02-0.03, Raised: >0.04
        RAISE_THRESHOLD = 0.035
        
        is_raised = avg_distance > RAISE_THRESHOLD
        
        # Confidence based on how much above threshold
        if is_raised:
            confidence = min(0.9, 0.5 + (avg_distance - RAISE_THRESHOLD) * 10)
        else:
            confidence = 0.0
        
        return is_raised, confidence
    
    def draw_landmarks(
        self,
        frame: np.ndarray,
        result: GestureResult,
        draw_face: bool = False,
        draw_eyebrows_only: bool = True,
    ) -> np.ndarray:
        """
        Draw hand and optionally face landmarks on the frame for visualization.
        
        Args:
            frame: BGR image
            result: GestureResult from detect()
            draw_face: Whether to draw face landmarks
            draw_eyebrows_only: If True, only draw eyebrow landmarks (less cluttered)
            
        Returns:
            Frame with landmarks drawn
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw hand landmarks
        for hand_kp in result.hand_keypoints:
            # Draw all 21 hand landmarks
            for i, (x, y, z) in enumerate(hand_kp.landmarks):
                # Convert normalized coordinates to pixel coordinates
                px, py = int(x * w), int(y * h)
                
                # Draw landmark point
                color = (0, 255, 0) if hand_kp.hand_label == 'Left' else (255, 0, 0)
                cv2.circle(annotated, (px, py), 5, color, -1)
                
                # Draw landmark number for debugging
                cv2.putText(
                    annotated,
                    str(i),
                    (px + 7, py - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1,
                )
            
            # Draw connections between landmarks
            self._draw_hand_connections(annotated, hand_kp.landmarks, w, h, hand_kp.hand_label)
        
        # Draw face landmarks if requested
        if draw_face and result.face_landmarks:
            self._draw_face_landmarks(annotated, result.face_landmarks, w, h, draw_eyebrows_only)
        
        return annotated
    
    def _draw_face_landmarks(
        self,
        frame: np.ndarray,
        face_landmarks,
        w: int,
        h: int,
        eyebrows_only: bool = True,
    ):
        """Draw face landmarks with emphasis on eyebrows."""
        if eyebrows_only:
            # Draw only eyebrow and eye landmarks for clarity
            # Right eyebrow landmarks (indices 70, 63, 105, 66, 107)
            right_eyebrow_indices = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
            # Left eyebrow landmarks (indices 300, 293, 334, 296, 336)
            left_eyebrow_indices = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
            # Right eye landmarks
            right_eye_indices = [33, 160, 158, 133, 153, 144, 163, 7, 159]
            # Left eye landmarks  
            left_eye_indices = [362, 385, 387, 263, 373, 380, 374, 249, 386]
            # Chin and mouth landmarks for thinking gesture
            chin_mouth_indices = [152, 13, 14, 17, 0, 61, 291]  # Chin, lips, mouth center
            
            # Draw eyebrow landmarks
            for idx in right_eyebrow_indices:
                lm = face_landmarks.landmark[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)  # Yellow for eyebrows
            
            for idx in left_eyebrow_indices:
                lm = face_landmarks.landmark[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)  # Yellow for eyebrows
            
            # Draw eye landmarks
            for idx in right_eye_indices:
                lm = face_landmarks.landmark[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 2, (255, 0, 255), -1)  # Magenta for eyes
            
            for idx in left_eye_indices:
                lm = face_landmarks.landmark[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 2, (255, 0, 255), -1)  # Magenta for eyes
            
            # Draw chin/mouth landmarks
            for idx in chin_mouth_indices:
                lm = face_landmarks.landmark[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 4, (255, 255, 0), -1)  # Cyan for chin/mouth
            
            # Draw connections between eyebrow points
            for i in range(len(right_eyebrow_indices) - 1):
                start_idx = right_eyebrow_indices[i]
                end_idx = right_eyebrow_indices[i + 1]
                start_lm = face_landmarks.landmark[start_idx]
                end_lm = face_landmarks.landmark[end_idx]
                start_px = (int(start_lm.x * w), int(start_lm.y * h))
                end_px = (int(end_lm.x * w), int(end_lm.y * h))
                cv2.line(frame, start_px, end_px, (0, 200, 200), 1)
            
            for i in range(len(left_eyebrow_indices) - 1):
                start_idx = left_eyebrow_indices[i]
                end_idx = left_eyebrow_indices[i + 1]
                start_lm = face_landmarks.landmark[start_idx]
                end_lm = face_landmarks.landmark[end_idx]
                start_px = (int(start_lm.x * w), int(start_lm.y * h))
                end_px = (int(end_lm.x * w), int(end_lm.y * h))
                cv2.line(frame, start_px, end_px, (0, 200, 200), 1)
        else:
            # Draw all 468 face landmarks (will be very dense)
            for i, lm in enumerate(face_landmarks.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)
    
    def _draw_hand_connections(
        self,
        frame: np.ndarray,
        landmarks: list[tuple[float, float, float]],
        w: int,
        h: int,
        hand_label: str,
    ):
        """Draw connections between hand landmarks."""
        # MediaPipe hand connections
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17),
        ]
        
        color = (0, 200, 0) if hand_label == 'Left' else (200, 0, 0)
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                start_px = (int(start[0] * w), int(start[1] * h))
                end_px = (int(end[0] * w), int(end[1] * h))
                
                cv2.line(frame, start_px, end_px, color, 2)
    
    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'holistic'):
            self.holistic.close()
            logger.info("GestureDetector closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def gesture_overlap_score(
    user_gestures: Sequence[str],
    meme_gestures: Sequence[str],
) -> float:
    """
    Calculate overlap between user and meme gesture tags.
    
    Args:
        user_gestures: List of detected user gesture tags
        meme_gestures: List of meme gesture tags from index
        
    Returns:
        Overlap score from 0.0 to 1.0
    """
    if not user_gestures or not meme_gestures:
        return 0.0
    
    user_set = set(user_gestures)
    meme_set = set(meme_gestures)
    
    # Jaccard similarity: intersection / union
    intersection = len(user_set & meme_set)
    union = len(user_set | meme_set)
    
    if union == 0:
        return 0.0
    
    return intersection / union
