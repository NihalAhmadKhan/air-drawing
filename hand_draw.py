"""
Hand Tracking Drawing Application
Draw on camera feed using your fingertip with gesture controls.
Uses MediaPipe Tasks API (newer version).
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions


class HandDrawingApp:
    def __init__(self):
        # Initialize MediaPipe Hand Landmarker
        model_path = self._download_model()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Canvas setup (transparent overlay)
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Drawing state
        self.current_color = (0, 0, 255)  # Default red (BGR)
        self.brush_size = 8
        self.eraser_size = 50  # Increased for better erasing
        self.is_drawing = False
        self.is_erasing = False
        self.prev_x, self.prev_y = None, None

        # Position smoothing
        self.position_history = deque(maxlen=8)  # Increased for more smoothing
        self.smoothing_enabled = True
        self.max_jump_distance = 50  # Ignore jumps larger than this (pixels)

        # Gesture state
        self.gesture_cooldown = 0
        self.gesture_cooldown_max = 15
        self.last_gesture = None

        # Color selection mode
        self.color_select_mode = False
        self.hover_start_time = None
        self.hover_color_idx = None
        self.hover_delay = 1.0  # Seconds to hover before selecting

        # Undo/Redo system
        self.undo_stack = deque(maxlen=20)
        self.redo_stack = deque(maxlen=20)
        self.save_state()

        # UI Colors
        self.colors = [
            ((0, 0, 255), "Red"),
            ((0, 255, 0), "Green"),
            ((255, 0, 0), "Blue"),
            ((0, 255, 255), "Yellow"),
            ((255, 0, 255), "Magenta"),
            ((255, 255, 0), "Cyan"),
            ((255, 255, 255), "White"),
            ((128, 128, 128), "Gray"),
        ]
        self.color_button_size = 50  # Slightly larger for easier selection
        self.ui_height = 70

        # Finger landmark indices
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        self.WRIST = 0

    def _download_model(self):
        """Download the hand landmarker model if not present"""
        import os
        import urllib.request

        model_dir = os.path.expanduser("~/.mediapipe/models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "hand_landmarker.task")

        if not os.path.exists(model_path):
            print("Downloading hand landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print(f"Model downloaded to {model_path}")

        return model_path

    def save_state(self):
        """Save current canvas state for undo/redo"""
        self.undo_stack.append(self.canvas.copy())
        self.redo_stack.clear()

    def undo(self):
        """Undo last action"""
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self.canvas = self.undo_stack[-1].copy()

    def redo(self):
        """Redo last undone action"""
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.undo_stack.append(state)
            self.canvas = state.copy()

    def clear_canvas(self):
        """Clear the canvas"""
        self.save_state()
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def get_finger_tip(self, landmarks, finger_idx):
        """Get finger tip coordinates"""
        landmark = landmarks[finger_idx]
        x = int(landmark.x * self.width)
        y = int(landmark.y * self.height)
        return x, y

    def get_smoothed_position(self, x, y):
        """Apply smoothing and outlier rejection to finger position"""
        if not self.smoothing_enabled:
            return x, y

        # Outlier rejection - ignore sudden large jumps
        if self.position_history:
            last_x, last_y = self.position_history[-1]
            distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
            if distance > self.max_jump_distance and len(self.position_history) > 3:
                # This is a jump, use last position instead
                return last_x, last_y

        self.position_history.append((x, y))
        if len(self.position_history) < 3:
            return x, y

        # Weighted average - more recent positions have higher weight
        weights = np.linspace(0.5, 1.0, len(self.position_history))
        weights /= weights.sum()

        avg_x = int(sum(pos[0] * w for pos, w in zip(self.position_history, weights)))
        avg_y = int(sum(pos[1] * w for pos, w in zip(self.position_history, weights)))

        return avg_x, avg_y

    def get_finger_states(self, landmarks):
        """
        Determine which fingers are extended
        Returns a dict with finger states
        """
        fingers = {}

        # Thumb (check x position for left/right hand)
        if landmarks[self.THUMB_TIP].x < landmarks[self.THUMB_TIP - 1].x:
            fingers['thumb'] = landmarks[self.THUMB_TIP].x < landmarks[self.THUMB_TIP - 2].x
        else:
            fingers['thumb'] = landmarks[self.THUMB_TIP].x > landmarks[self.THUMB_TIP - 2].x

        # Other fingers (check if tip is above PIP joint - y is inverted in image coordinates)
        fingers['index'] = landmarks[self.INDEX_TIP].y < landmarks[self.INDEX_TIP - 2].y
        fingers['middle'] = landmarks[self.MIDDLE_TIP].y < landmarks[self.MIDDLE_TIP - 2].y
        fingers['ring'] = landmarks[self.RING_TIP].y < landmarks[self.RING_TIP - 2].y
        fingers['pinky'] = landmarks[self.PINKY_TIP].y < landmarks[self.PINKY_TIP - 2].y

        return fingers

    def detect_gesture(self, landmarks):
        """
        Detect hand gestures for controls
        Returns: gesture name or None
        """
        fingers = self.get_finger_states(landmarks)
        count = sum(fingers.values())

        # Gesture definitions based on finger count and specific combinations
        if count == 0:
            return "fist"  # Fist = erase mode
        elif fingers.get('index') and not fingers.get('middle') and not fingers.get('ring') and not fingers.get('pinky'):
            return "point"  # Only index = draw mode
        elif count == 5:
            return "palm"  # Open palm = clear canvas
        elif fingers.get('index') and fingers.get('middle') and not fingers.get('ring') and not fingers.get('pinky'):
            return "peace"  # Index + middle = undo
        elif fingers.get('thumb') and fingers.get('index') and not fingers.get('middle'):
            return "pinch"  # Thumb + index = redo
        elif count == 3:
            return "three"  # Three fingers = color select mode

        return None

    def get_color_at_position(self, x, y):
        """Get color index if position is in color palette"""
        if y > self.height - self.ui_height:
            color_idx = x // self.color_button_size
            if 0 <= color_idx < len(self.colors):
                return color_idx
        return None

    def draw_ui(self, frame):
        """Draw UI elements on frame"""
        # Semi-transparent bottom bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, self.height - self.ui_height),
                      (self.width, self.height), (50, 50, 50), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Draw color buttons
        for i, (color, name) in enumerate(self.colors):
            x = i * self.color_button_size
            y = self.height - self.ui_height

            # Draw button
            cv2.rectangle(frame, (x + 2, y + 2),
                         (x + self.color_button_size - 2, y + self.color_button_size - 2), color, -1)

            # Highlight selected color
            if color == self.current_color and not self.is_erasing:
                cv2.rectangle(frame, (x, y),
                             (x + self.color_button_size, y + self.color_button_size), (255, 255, 255), 3)

            # Highlight hover color with progress
            if self.hover_color_idx == i and self.color_select_mode:
                elapsed = cv2.getTickCount() - self.hover_start_time if self.hover_start_time else 0
                progress = min(1.0, elapsed / (self.hover_delay * cv2.getTickFrequency()))
                bar_width = int((self.color_button_size - 4) * progress)
                cv2.rectangle(frame, (x + 2, y + self.color_button_size - 8),
                             (x + 2 + bar_width, y + self.color_button_size - 4), (0, 255, 0), -1)

            # Border
            cv2.rectangle(frame, (x, y),
                         (x + self.color_button_size, y + self.color_button_size), (100, 100, 100), 1)

            # Color name
            text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
            cv2.putText(frame, name[:3], (x + 8, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

        # Draw mode indicator
        mode_x = len(self.colors) * self.color_button_size + 20
        mode_y = self.height - self.ui_height + 15

        if self.is_erasing:
            cv2.putText(frame, "ERASE MODE", (mode_x, mode_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # Show eraser size indicator
            cv2.putText(frame, f"Size: {self.eraser_size}px", (mode_x, mode_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            color_name = next((name for c, name in self.colors if c == self.current_color), "Draw")
            cv2.putText(frame, f"DRAW: {color_name}", (mode_x, mode_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.current_color, 2)
            cv2.putText(frame, f"Size: {self.brush_size}px", (mode_x, mode_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Gesture indicator
        if self.last_gesture:
            gesture_text = f"Gesture: {self.last_gesture.upper()}"
            cv2.putText(frame, gesture_text, (mode_x, mode_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Instructions
        instructions = [
            "Point=Draw | Fist=Erase | Peace=Undo | Pinch=Redo | Palm=Clear | Hover+Pinch=Color",
        ]
        for i, instr in enumerate(instructions):
            cv2.putText(frame, instr, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def draw_line(self, x1, y1, x2, y2):
        """Draw a line on canvas"""
        if self.is_erasing:
            color = (0, 0, 0)
            size = self.eraser_size
        else:
            color = self.current_color
            size = self.brush_size
        cv2.line(self.canvas, (x1, y1), (x2, y2), color, size, cv2.LINE_AA)

    def run(self):
        """Main application loop"""
        print("Hand Drawing App Started!")
        print("Controls:")
        print("  - Point with index finger: Draw")
        print("  - Make fist: Erase")
        print("  - Peace sign (index+middle): Undo")
        print("  - Pinch (thumb+index): Redo")
        print("  - Open palm: Clear canvas")
        print("  - Hover over color + Pinch to select color")
        print("  - Press 'q' or ESC to exit")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)

            # Get index finger tip for drawing
            index_x, index_y = None, None
            gesture = None

            if results.hand_landmarks:
                landmarks = results.hand_landmarks[0]

                # Detect gesture
                gesture = self.detect_gesture(landmarks)

                # Get index finger tip position
                raw_x, raw_y = self.get_finger_tip(landmarks, self.INDEX_TIP)

                # Apply smoothing
                index_x, index_y = self.get_smoothed_position(raw_x, raw_y)

                # Draw hand landmarks on frame
                self._draw_landmarks(frame, landmarks)

                # Draw finger tip indicator with different color based on mode
                if self.is_erasing:
                    # Draw eraser indicator
                    cv2.circle(frame, (index_x, index_y), self.eraser_size // 2, (0, 255, 255), 2)
                    cv2.circle(frame, (index_x, index_y), 5, (0, 255, 255), -1)
                else:
                    # Draw brush indicator
                    cv2.circle(frame, (index_x, index_y), self.brush_size // 2 + 2, self.current_color, 2)
                    cv2.circle(frame, (index_x, index_y), 5, (255, 255, 0), -1)

                # Handle gesture cooldown
                if self.gesture_cooldown > 0:
                    self.gesture_cooldown -= 1

                # Check if in color palette area
                color_idx = self.get_color_at_position(index_x, index_y)

                # Process gestures
                if gesture and self.gesture_cooldown == 0:
                    self.last_gesture = gesture

                    if gesture == "fist":
                        self.is_erasing = True
                        self.is_drawing = False
                        self.prev_x, self.prev_y = None, None
                    elif gesture == "point":
                        self.is_erasing = False
                        self.is_drawing = True
                    elif gesture == "peace":
                        self.undo()
                        self.gesture_cooldown = self.gesture_cooldown_max
                        self.is_drawing = False
                        self.prev_x, self.prev_y = None, None
                    elif gesture == "pinch":
                        # If hovering over color, select it
                        if color_idx is not None:
                            self.current_color = self.colors[color_idx][0]
                            self.is_erasing = False
                            self.gesture_cooldown = self.gesture_cooldown_max
                        else:
                            self.redo()
                            self.gesture_cooldown = self.gesture_cooldown_max
                        self.is_drawing = False
                        self.prev_x, self.prev_y = None, None
                    elif gesture == "palm":
                        self.clear_canvas()
                        self.gesture_cooldown = self.gesture_cooldown_max
                        self.is_drawing = False
                        self.prev_x, self.prev_y = None, None

                # Handle color hover selection
                if color_idx is not None:
                    if self.hover_color_idx != color_idx:
                        self.hover_color_idx = color_idx
                        self.hover_start_time = cv2.getTickCount()
                        self.color_select_mode = True
                    elif self.color_select_mode:
                        # Check if hovered long enough and pinch gesture
                        elapsed = (cv2.getTickCount() - self.hover_start_time) / cv2.getTickFrequency()
                        if elapsed >= self.hover_delay and gesture == "pinch":
                            self.current_color = self.colors[color_idx][0]
                            self.is_erasing = False
                            self.color_select_mode = False
                            self.gesture_cooldown = self.gesture_cooldown_max
                else:
                    self.hover_color_idx = None
                    self.color_select_mode = False

                # Handle drawing/erasing
                if self.is_drawing or self.is_erasing:
                    if index_x is not None and index_y is not None:
                        # Check if in UI area - don't draw in UI
                        if index_y < self.height - self.ui_height:
                            if self.prev_x is not None and self.prev_y is not None:
                                self.draw_line(self.prev_x, self.prev_y, index_x, index_y)
                            self.prev_x, self.prev_y = index_x, index_y
                        else:
                            self.prev_x, self.prev_y = None, None
                else:
                    self.prev_x, self.prev_y = None, None
            else:
                self.is_drawing = False
                self.prev_x, self.prev_y = None, None
                self.hover_color_idx = None
                self.color_select_mode = False
                self.position_history.clear()

            # Save state when drawing stops
            if not self.is_drawing and self.prev_x is not None:
                self.save_state()
                self.prev_x, self.prev_y = None, None

            # Merge canvas with frame
            gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)

            frame = cv2.add(frame_bg, canvas_fg)

            # Draw UI
            frame = self.draw_ui(frame)

            # Display
            cv2.imshow("Hand Drawing App", frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('u'):  # Manual undo
                self.undo()
            elif key == ord('r'):  # Manual redo
                self.redo()
            elif key == ord('c'):  # Manual clear
                self.clear_canvas()
            elif key == ord('e'):  # Toggle eraser
                self.is_erasing = not self.is_erasing
                self.is_drawing = not self.is_erasing
            elif key == ord('+') or key == ord('='):  # Increase brush/eraser size
                if self.is_erasing:
                    self.eraser_size = min(100, self.eraser_size + 5)
                else:
                    self.brush_size = min(50, self.brush_size + 2)
            elif key == ord('-'):  # Decrease brush/eraser size
                if self.is_erasing:
                    self.eraser_size = max(20, self.eraser_size - 5)
                else:
                    self.brush_size = max(2, self.brush_size - 2)

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

    def _draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame"""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        h, w = frame.shape[:2]

        for start, end in connections:
            x1 = int(landmarks[start].x * w)
            y1 = int(landmarks[start].y * h)
            x2 = int(landmarks[end].x * w)
            y2 = int(landmarks[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for i, lm in enumerate(landmarks):
            x = int(lm.x * w)
            y = int(lm.y * h)
            color = (0, 255, 0) if i == self.INDEX_TIP else (255, 0, 0)
            cv2.circle(frame, (x, y), 5, color, -1)


if __name__ == "__main__":
    app = HandDrawingApp()
    app.run()
