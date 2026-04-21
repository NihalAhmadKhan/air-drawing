# Hand Tracking Drawing Application

A computer vision project that allows you to draw on a canvas using your hand gestures. Built with OpenCV and MediaPipe.

## Features

- **Hand Tracking**: Real-time hand tracking using MediaPipe Hands
- **Smooth Drawing**: Position smoothing and outlier rejection to prevent flickering
- **Freehand Drawing**: Draw lines using your index finger
- **Eraser Mode**: Erase parts of your drawing using fist gesture (large eraser size)
- **Color Selection**: Hover over color + pinch to select, or pinch while hovering
- **Undo/Redo**: Full undo/redo history (gesture controlled)
- **Clear Canvas**: Clear the entire canvas with a gesture
- **Adjustable Brush/Eraser Size**: Use +/- keys to adjust

## Gestures

| Gesture | Description | Action |
|---------|-------------|--------|
| Point (Index only) | Extend only index finger | Draw mode |
| Fist | Close all fingers | Erase mode |
| Peace Sign | Index + Middle fingers | Undo |
| Pinch | Thumb + Index fingers | Redo (or select color when hovering) |
| Open Palm | All fingers extended | Clear canvas |

## Color Selection

To select a color:
1. Move your finger over the color palette at the bottom
2. Wait for the green bar to fill up (auto-select after ~0.8 seconds)
3. **OR** pinch (thumb+index) while hovering to select immediately

The currently selected color is highlighted with a white border.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
# Basic version
python hand_draw.py

# Advanced version (with enhanced smoothing and additional features)
python hand_draw_advanced.py
```

## Usage

### Drawing Tips

- **Move slowly** for smoother lines - the smoothing algorithm works best with steady motion
- **Keep your hand visible** - if hand tracking is lost, drawing pauses
- **Use eraser mode** (fist) for quick corrections

### Basic Controls (hand_draw.py)

**Gestures:**
- Point with index finger to draw
- Make a fist to erase
- Show peace sign (✌️) to undo
- Pinch thumb and index to redo (or select color when hovering)
- Open palm to clear canvas

**Keyboard Shortcuts:**
- `Q` or `ESC` - Exit
- `U` - Undo
- `R` - Redo
- `C` - Clear canvas
- `E` - Toggle eraser mode
- `+` / `-` - Increase/decrease brush/eraser size

### Advanced Controls (hand_draw_advanced.py)

**Additional Keyboard Shortcuts:**
- `+` / `-` - Adjust brush/eraser size
- `S` - Toggle smoothing
- `H` - Toggle UI visibility
- `U` / `R` - Undo/Redo

## Files

- `hand_draw.py` - Basic drawing application with smoothing
- `hand_draw_advanced.py` - Enhanced version with advanced smoothing and features
- `requirements.txt` - Python dependencies
- `CLAUDE.md` - Developer documentation

## Requirements

- Python 3.7+
- Webcam
- Dependencies:
  - opencv-python
  - mediapipe
  - numpy

## Troubleshooting

### Tracking Issues / Flickering

If the drawing point jumps around:
- **Enable smoothing** (press `S` in advanced version)
- Move your hand more slowly and steadily
- Ensure good, even lighting
- Keep your hand fully visible in the frame

### Color Selection Not Working

- Make sure your finger is clearly over the color square
- Wait for the green bar to fill, or pinch while hovering
- If using pinch, make sure thumb and index finger are clearly touching

### Camera Not Opening

- Check if another application is using the camera
- Try changing the camera index in the code (`cv2.VideoCapture(0)`)

### Hand Not Detected

- Ensure your hand is fully visible in the frame
- Adjust lighting conditions
- Try moving your hand closer to or further from the camera

### Lag/Low FPS

- Close other applications using CPU/GPU
- Reduce camera resolution (modify `width` and `height` in the code)
- Consider using the basic version instead of advanced

## Tips for Best Results

1. **Lighting**: Ensure good, even lighting on your hand
2. **Background**: Use a plain background if possible
3. **Distance**: Keep your hand about 1-2 feet from the camera
4. **Speed**: Move your finger smoothly at moderate speed
5. **Smoothing**: Both versions now have smoothing enabled by default for cleaner curves

## Technical Details

### Smoothing Algorithm

Both versions use:
- **Position history buffer** (8-10 frames) with weighted averaging
- **Outlier rejection** - ignores sudden jumps > 40-50 pixels
- **Velocity prediction** - predicts position based on movement trend
- **Exponential moving average** for smooth transitions

### Eraser

- Eraser size: 50px (increased from 30px for better efficiency)
- Visual indicator shows eraser area
- Adjustable with +/- keys

## License

This project is open source and available for personal and educational use.

