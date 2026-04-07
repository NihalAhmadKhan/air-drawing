"""
Advanced Hand Tracking Drawing Application
Enhanced version with additional features like brush size control,
palette selection, and smoother drawing experience.
Uses MediaPipe Tasks API (newer version).
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from enum import Enum
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions


class DrawingMode(Enum):
    DRAW = "draw"
    ERASE = "erase"


class HandDrawingAppAdvanced:
    def __init__(self):
        # Initialize MediaPipe Hand Landmarker
        model_path = self._download_model()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Canvas setup with layers
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Drawing state
        self.current_color = (0, 0, 255)
        self.brush_size = 8
        self.eraser_size = 50  # Increased for better erasing
        self.drawing_mode = DrawingMode.DRAW
        self.prev_x, self.prev_y = None, None

        # Enhanced smoothing
        self.position_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        self.smoothing_enabled = True
        self.max_jump_distance = 40
        self.prediction_factor = 0.3

        # Gesture state
        self.gesture_cooldown = 0
        self.gesture_cooldown_max = 12
        self.last_gesture = None
        self.last_gesture_time = 0

        # Color selection with hover
        self.color_select_mode = False
        self.hover_start_time = None
        self.hover_color_idx = None
        self.hover_delay = 0.8

        # Undo/Redo system
        self.undo_stack = deque(maxlen=30)
        self.redo_stack = deque(maxlen=30)
        self.save_state()

        # UI
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
        self.color_button_size = 55
        self.ui_height = 100
        self.ui_visible = True

        # Finger landmarks
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20

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
        """Save current canvas state"""
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
        """Apply enhanced smoothing with outlier rejection and velocity prediction"""
        if not self.smoothing_enabled:
            return x, y

        # Outlier rejection
        if self.position_history:
            last_x, last_y = self.position_history[-1]
            distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)

            if distance > self.max_jump_distance and len(self.position_history) > 3:
                # Sudden jump detected, interpolate instead
                velocity_x = self.velocity_history[-1][0] if self.velocity_history else 0
                velocity_y = self.velocity_history[-1][1] if self.velocity_history else 0

                predicted_x = int(last_x + velocity_x * self.prediction_factor)
                predicted_y = int(last_y + velocity_y * self.prediction_factor)

                # Use predicted position instead of jump
                x, y = predicted_x, predicted_y

        # Calculate velocity
        if self.position_history:
            last_x, last_y = self.position_history[-1]
            self.velocity_history.append((x - last_x, y - last_y))

        self.position_history.append((x, y))

        if len(self.position_history) < 3:
            return x, y

        # Exponential moving average with more weight on recent positions
        alpha = 0.4
        smoothed_x = self.position_history[-1][0]
        smoothed_y = self.position_history[-1][1]

        for i in range(len(self.position_history) - 2, -1, -1):
            smoothed_x = alpha * self.position_history[i][0] + (1 - alpha) * smoothed_x
            smoothed_y = alpha * self.position_history[i][1] + (1 - alpha) * smoothed_y

        return int(smoothed_x), int(smoothed_y)

    def get_finger_states(self, landmarks):
        """Determine which fingers are extended"""
        fingers = {}

        # Thumb
        fingers['thumb'] = landmarks[self.THUMB_TIP].x < landmarks[self.THUMB_TIP - 2].x

        # Other fingers
        fingers['index'] = landmarks[self.INDEX_TIP].y < landmarks[self.INDEX_TIP - 2].y
        fingers['middle'] = landmarks[self.MIDDLE_TIP].y < landmarks[self.MIDDLE_TIP - 2].y
        fingers['ring'] = landmarks[self.RING_TIP].y < landmarks[self.RING_TIP - 2].y
        fingers['pinky'] = landmarks[self.PINKY_TIP].y < landmarks[self.PINKY_TIP - 2].y

        return fingers

    def detect_gesture(self, landmarks):
        """Detect hand gestures for controls"""
        fingers = self.get_finger_states(landmarks)
        count = sum(fingers.values())

        if count == 0:
            return "fist"
        elif fingers.get('index') and not fingers.get('middle'):
            return "point"
        elif fingers.get('index') and fingers.get('middle') and not fingers.get('ring'):
            return "two"
        elif count == 3:
            return "three"
        elif count == 5:
            return "palm"
        elif fingers.get('thumb') and fingers.get('index'):
            return "pinch"

        return None

    def get_color_at_position(self, x, y):
        """Get color index if position is in color palette"""
        if y > self.height - self.ui_height and y < self.height - self.ui_height + 55:
            color_idx = x // self.color_button_size
            if 0 <= color_idx < len(self.colors):
                return color_idx
        return None

    def draw_ui(self, frame):
        """Draw UI elements on frame"""
        if not self.ui_visible:
            return frame

        # UI background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, self.height - self.ui_height),
                      (self.width, self.height), (40, 40, 40), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        # Color buttons with hover indicator
        for i, (color, name) in enumerate(self.colors):
            x = i * self.color_button_size + 10
            y = self.height - self.ui_height + 10
            btn_size = self.color_button_size - 10

            # Draw button
            cv2.rectangle(frame, (x, y), (x + btn_size, y + btn_size), color, -1)

            # Highlight selected color
            if color == self.current_color and self.drawing_mode == DrawingMode.DRAW:
                cv2.rectangle(frame, (x - 2, y - 2), (x + btn_size + 2, y + btn_size + 2), (255, 255, 255), 3)

            # Highlight hover color
            if self.hover_color_idx == i:
                # Draw selection progress
                elapsed = (cv2.getTickCount() - self.hover_start_time) / cv2.getTickFrequency() if self.hover_start_time else 0
                progress = min(1.0, elapsed / self.hover_delay)
                bar_height = int(btn_size * progress)
                cv2.rectangle(frame, (x, y + btn_size - bar_height), (x + btn_size, y + btn_size), (0, 255, 0), -1)
                cv2.rectangle(frame, (x - 2, y - 2), (x + btn_size + 2, y + btn_size + 2), (0, 255, 0), 2)

            # Border
            cv2.rectangle(frame, (x, y), (x + btn_size, y + btn_size), (100, 100, 100), 1)

            # Color name
            text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
            cv2.putText(frame, name[:3], (x + 8, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

        # Mode buttons
        modes_y = self.height - self.ui_height + 70
        mode_x = 10

        # Draw button
        color = (100, 200, 100) if self.drawing_mode == DrawingMode.DRAW else (80, 80, 80)
        cv2.rectangle(frame, (mode_x, modes_y), (mode_x + 60, modes_y + 25), color, -1)
        cv2.putText(frame, "Draw", (mode_x + 10, modes_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Erase button
        mode_x += 70
        color = (100, 200, 100) if self.drawing_mode == DrawingMode.ERASE else (80, 80, 80)
        cv2.rectangle(frame, (mode_x, modes_y), (mode_x + 60, modes_y + 25), color, -1)
        cv2.putText(frame, "Erase", (mode_x + 5, modes_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Undo/Redo/Clear buttons
        mode_x += 80
        cv2.rectangle(frame, (mode_x, modes_y), (mode_x + 50, modes_y + 25), (80, 80, 200), -1)
        cv2.putText(frame, "Undo", (mode_x + 5, modes_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        mode_x += 60
        cv2.rectangle(frame, (mode_x, modes_y), (mode_x + 50, modes_y + 25), (80, 80, 200), -1)
        cv2.putText(frame, "Redo", (mode_x + 5, modes_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        mode_x += 60
        cv2.rectangle(frame, (mode_x, modes_y), (mode_x + 50, modes_y + 25), (200, 80, 80), -1)
        cv2.putText(frame, "Clear", (mode_x + 5, modes_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Info panel
        info_x = self.width - 220
        cv2.putText(frame, f"Brush: {self.brush_size}px", (info_x, self.height - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Eraser: {self.eraser_size}px", (info_x, self.height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        mode_text = "ERASE" if self.drawing_mode == DrawingMode.ERASE else "DRAW"
        mode_color = (0, 255, 255) if self.drawing_mode == DrawingMode.ERASE else self.current_color
        cv2.putText(frame, f"Mode: {mode_text}", (info_x, self.height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        return frame

    def handle_ui_interaction(self, x, y, gesture):
        """Handle UI hover and selection"""
        if y < self.height - self.ui_height:
            self.hover_color_idx = None
            self.color_select_mode = False
            return False

        # Check color hover
        color_idx = self.get_color_at_position(x, y)
        if color_idx is not None:
            if self.hover_color_idx != color_idx:
                self.hover_color_idx = color_idx
                self.hover_start_time = cv2.getTickCount()
                self.color_select_mode = True
            elif gesture == "pinch":
                # Select color on pinch
                self.current_color = self.colors[color_idx][0]
                self.drawing_mode = DrawingMode.DRAW
                self.color_select_mode = False
                return True
            elif self.color_select_mode:
                # Check for auto-select after hover delay
                elapsed = (cv2.getTickCount() - self.hover_start_time) / cv2.getTickFrequency()
                if elapsed >= self.hover_delay:
                    self.current_color = self.colors[color_idx][0]
                    self.drawing_mode = DrawingMode.DRAW
                    self.color_select_mode = False
                    return True
            return True
        else:
            self.hover_color_idx = None
            self.color_select_mode = False

        # Check button clicks (y area)
        if self.height - self.ui_height + 70 <= y <= self.height - self.ui_height + 95:
            # Draw button
            if 10 <= x <= 70:
                self.drawing_mode = DrawingMode.DRAW
                return True
            # Erase button
            elif 80 <= x <= 140:
                self.drawing_mode = DrawingMode.ERASE
                return True
            # Undo button
            elif 160 <= x <= 210:
                self.undo()
                return True
            # Redo button
            elif 220 <= x <= 270:
                self.redo()
                return True
            # Clear button
            elif 280 <= x <= 330:
                self.clear_canvas()
                return True

        return False

    def draw_line(self, x1, y1, x2, y2):
        """Draw a line on canvas"""
        if self.drawing_mode == DrawingMode.ERASE:
            color = (0, 0, 0)
            size = self.eraser_size
        else:
            color = self.current_color
            size = self.brush_size
        cv2.line(self.canvas, (x1, y1), (x2, y2), color, size, cv2.LINE_AA)

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

    def run(self):
        """Main application loop"""
        print("Advanced Hand Drawing App Started!")
        print("\nControls:")
        print("  Point (index only): Draw")
        print("  Fist (all closed): Erase")
        print("  Peace (index+middle): Undo")
        print("  Pinch (thumb+index): Redo / Select Color")
        print("  Palm (all open): Clear canvas")
        print("  Hover over color + Pinch: Select that color")
        print("\nKeyboard:")
        print("  +/- : Brush size | S: Toggle smoothing")
        print("  H: Toggle UI | U/R: Undo/Redo | C: Clear")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)

            index_x, index_y = None, None
            gesture = None

            if results.hand_landmarks:
                landmarks = results.hand_landmarks[0]
                gesture = self.detect_gesture(landmarks)

                raw_x, raw_y = self.get_finger_tip(landmarks, self.INDEX_TIP)
                index_x, index_y = self.get_smoothed_position(raw_x, raw_y)

                # Draw hand landmarks
                self._draw_landmarks(frame, landmarks)

                # Draw cursor with visual feedback
                if self.drawing_mode == DrawingMode.ERASE:
                    cv2.circle(frame, (index_x, index_y), self.eraser_size // 2, (0, 255, 255), 2)
                    cv2.circle(frame, (index_x, index_y), 3, (0, 255, 255), -1)
                    cv2.putText(frame, "ERASE", (index_x + 20, index_y - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                else:
                    cv2.circle(frame, (index_x, index_y), self.brush_size // 2 + 2, self.current_color, 2)
                    cv2.circle(frame, (index_x, index_y), 3, (255, 255, 0), -1)

                # Handle gesture cooldown
                if self.gesture_cooldown > 0:
                    self.gesture_cooldown -= 1

                # Process gestures
                if gesture and self.gesture_cooldown == 0:
                    self.last_gesture = gesture

                    if gesture == "fist":
                        self.drawing_mode = DrawingMode.ERASE
                        self.prev_x, self.prev_y = None, None
                    elif gesture == "point":
                        self.drawing_mode = DrawingMode.DRAW
                    elif gesture == "two":
                        self.undo()
                        self.gesture_cooldown = self.gesture_cooldown_max
                        self.prev_x, self.prev_y = None, None
                    elif gesture == "pinch":
                        self.redo()
                        self.gesture_cooldown = self.gesture_cooldown_max
                        self.prev_x, self.prev_y = None, None
                    elif gesture == "palm":
                        self.clear_canvas()
                        self.gesture_cooldown = self.gesture_cooldown_max
                        self.prev_x, self.prev_y = None, None

                # Handle UI interactions and drawing
                if index_x is not None and index_y is not None:
                    if not self.handle_ui_interaction(index_x, index_y, gesture):
                        # Drawing area - only draw with point or fist gesture
                        if gesture in ["point", "fist"]:
                            if self.prev_x is not None and self.prev_y is not None:
                                self.draw_line(self.prev_x, self.prev_y, index_x, index_y)
                            self.prev_x, self.prev_y = index_x, index_y
                        else:
                            self.prev_x, self.prev_y = None, None
                    else:
                        self.gesture_cooldown = self.gesture_cooldown_max
                        self.prev_x, self.prev_y = None, None
            else:
                self.prev_x, self.prev_y = None, None
                self.hover_color_idx = None
                self.color_select_mode = False
                self.position_history.clear()
                self.velocity_history.clear()

            # Save state when drawing stops
            if self.prev_x is not None and (index_x is None or gesture not in ["point", "fist"]):
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

            # Status text
            cv2.putText(frame, "Hover+Pinch on color to select | +/- Size | S Smooth | H UI | U/R Undo/Redo",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Advanced Hand Drawing App", frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('u'):
                self.undo()
            elif key == ord('r'):
                self.redo()
            elif key == ord('c'):
                self.clear_canvas()
            elif key == ord('+') or key == ord('='):
                if self.drawing_mode == DrawingMode.ERASE:
                    self.eraser_size = min(100, self.eraser_size + 5)
                else:
                    self.brush_size = min(50, self.brush_size + 2)
            elif key == ord('-'):
                if self.drawing_mode == DrawingMode.ERASE:
                    self.eraser_size = max(20, self.eraser_size - 5)
                else:
                    self.brush_size = max(2, self.brush_size - 2)
            elif key == ord('s'):
                self.smoothing_enabled = not self.smoothing_enabled
                print(f"Smoothing: {'ON' if self.smoothing_enabled else 'OFF'}")
            elif key == ord('h'):
                self.ui_visible = not self.ui_visible

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = HandDrawingAppAdvanced()
    app.run()
