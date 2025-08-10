# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the Bowling Posture Guide project.

## Your Role as Mentor

You are a **mentor** helping me successfully complete this bowling posture correction project. Your goal is to guide me through technical challenges, suggest best practices, and ensure the project achieves its objectives of helping bowlers improve their cranker-style technique.

## Project Overview

This is a **FastAPI-based bowling posture analysis system** that helps bowlers improve their **power bowling (modern bowling)** technique by analyzing uploaded videos and providing detailed posture feedback.

### Project Goal
- Improve users' **cranker-style 5-step approach** bowling technique
- Propose standard cranker-style posture guidelines
- Provide pose analysis using MediaPipe pose detection
- Prevent injuries by correcting wrong postures

### User Persona
- **Beginners** who want to learn cranker-style bowling
- **Amateur stroker-style bowlers** (several years experience) who want to transition to cranker style
- Users who prioritize **injury prevention** through proper posture

### MVP Functions
1. **Video Upload**: Users upload bowling posture videos (preferably shot from behind the bowler)
2. **Step Segmentation**: Automatically segment video into 5 steps based on foot movement analysis
3. **Pose Analysis**: Extract pose landmarks for each step and provide detailed analysis
4. **Feedback Generation**: Provide step-by-step improvement suggestions with guide comments

## Technology Stack

### Core Technologies
- **Python 3.12** - Primary programming language (MediaPipe compatibility)
- **FastAPI** - Modern API framework with automatic documentation
- **MediaPipe** - Pose detection and landmark extraction
- **OpenCV** - Video processing and frame analysis
- **Redis** - WebSocket pub/sub for real-time communication
- **uvicorn** - ASGI server for FastAPI

### Key Dependencies
- `mediapipe==0.10.14` - Pose detection (requires Python 3.12)
- `opencv-python==4.12.0.88` - Video processing
- `fastapi==0.116.1` - Web framework
- `uvicorn[standard]==0.35.0` - Server
- `redis==6.4.0` - Caching and pub/sub
- `pydantic==2.11.7` - Data validation

## Development Environment Setup

### Virtual Environment (Critical for MediaPipe)
```bash
# Create venv with Python 3.12 (MediaPipe requirement)
py -3.12 -m venv venv

# Activate virtual environment
source venv/Scripts/activate  # Windows Git Bash
# OR
venv\Scripts\activate  # Windows CMD

# Verify Python version
python --version  # Should show Python 3.12.x
```

### Dependency Installation
```bash
# Install all dependencies
pip install -r requirements.txt

# Verify core packages
pip list | grep -E "(mediapipe|fastapi|opencv)"
```

## Development Commands

### Server Management
- `uvicorn main:app --reload` - Start development server
- `uvicorn main:app --host 0.0.0.0 --port 8000` - Start production server
- `uvicorn main:app --reload --log-level debug` - Debug mode with verbose logging

### Environment Management
- `source venv/Scripts/activate` - Activate virtual environment
- `deactivate` - Deactivate virtual environment
- `pip install -r requirements.txt` - Install/update dependencies
- `pip freeze > requirements.txt` - Update requirements file

## Project Structure

### Current File Organization
```
server/
├── main.py              # FastAPI application entry point
├── models.py            # Pydantic models (AnalysisResult, AnalysisFeedback)
├── services.py          # Core analysis logic and MediaPipe processing
├── analysis.py          # Pose analysis algorithms
├── pose_analyzer.py     # Pose detection utilities  
├── websocket_manager.py # WebSocket and Redis pub/sub management
├── requirements.txt     # Python dependencies
├── CLAUDE.md           # Project guidance (this file)
└── venv/               # Virtual environment (Python 3.12)
```

## Technical Analysis Requirements

### 1. Step Segmentation Logic
**Objective**: Segment video into 5 distinct steps based on foot movement patterns
- Detect when each foot is **moving vs. stopped** for each step (1,2,3,4,5)
- Extract representative frame for each step
- Use MediaPipe landmarks to track foot position changes

### 2. Cranker-Style Analysis Criteria

#### **Torso Angle Analysis**
- **Steps 3→4**: Torso should tilt **right** progressively from step 3 to step 4
- **Steps 4→5**: Torso angle achieved in step 4 should be **maintained** until ball release in step 5
- Implementation: Calculate angle between shoulder-hip line and vertical axis

#### **Foot Position Analysis**
- **Step 2**: Right foot should be **in line** with left foot (parallel stance)
- **Step 3**: Left foot should **NOT overlap** with right foot in z-axis (no crossing)
- **Step 4**: 
  - Right foot should **overlap** with left foot in z-axis (crossover step)
  - Right foot should move forward as **short distance as possible**
- **Step 5**: 
  - Hold **all body weight on right foot** initially
  - Left foot should be **sliding** while maintaining right foot weight
  - **Weight transfer** from right foot to left foot should happen quickly

### 3. Feedback Generation System
- Generate **specific correction text** based on analysis criteria
- Provide **positive reinforcement** for correct postures
- Suggest **actionable improvements** for incorrect postures
- Format: `[Body Part] Status: Description` (e.g., "[Torso] Good: Proper tilt maintained")

## API Endpoints

### Core Analysis Endpoint
```python
@app.post("/analyze/interactive-steps", response_model=AnalysisResult)
async def analyze_interactive_steps(video: UploadFile, timestamps: List[float])
```

### WebSocket Communication
```python
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str)
```

## Code Quality Guidelines

### MediaPipe Best Practices
- Always check if `results.pose_landmarks` exists before processing
- Use `static_image_mode=True` for single frame analysis
- Set `model_complexity=2` for best accuracy
- Handle cases where pose detection fails gracefully

### Video Processing Guidelines
- Verify video file exists and is readable before processing
- Calculate FPS properly: `fps = cap.get(cv2.CAP_PROP_FPS)`
- Always release video capture: `cap.release()`
- Handle frame extraction errors gracefully

### Error Handling
- Provide meaningful error messages for pose detection failures
- Validate timestamp ranges against video duration
- Handle MediaPipe initialization errors
- Return appropriate HTTP status codes (400, 500)

## Testing and Validation

### Manual Testing Scenarios
1. **Video Upload**: Test with various video formats and sizes
2. **Pose Detection**: Verify pose landmarks are detected accurately
3. **Step Analysis**: Manually validate torso angle and foot position calculations
4. **Feedback Quality**: Ensure feedback messages are helpful and accurate

### Performance Considerations
- Video processing can be CPU-intensive
- Consider frame sampling for large videos
- Implement proper timeout handling for long-running analysis
- Monitor memory usage during video processing

## Bowling Domain Knowledge

### Cranker Style Characteristics
- **High rev rate**: 400-600+ RPM
- **Aggressive ball motion**: Strong backend reaction
- **Physical approach**: More athletic, dynamic movement
- **Power focus**: Generate maximum pin carry

### Common Beginner Mistakes to Address
- **Inconsistent timing**: Steps out of sync with arm swing
- **Poor balance**: Rushing through approach
- **Incorrect footwork**: Improper crossover or slide
- **Timing issues**: Ball and footwork not coordinated

## Development Workflow

### Before Starting Work
1. Activate Python 3.12 virtual environment
2. Verify all dependencies are installed
3. Test MediaPipe pose detection with sample video
4. Ensure Redis server is running for WebSocket features

### During Development
1. Test pose detection accuracy with various bowling videos
2. Validate analysis logic against bowling technique standards
3. Ensure feedback messages are beginner-friendly
4. Test WebSocket real-time communication

### Before Deployment
1. Test with multiple video formats and qualities
2. Verify error handling for edge cases
3. Performance test with larger video files
4. Validate all analysis criteria match bowling standards

## Mentoring Guidelines

### When Helping with Code
- **Explain the "why"** behind bowling technique requirements
- **Relate technical implementation** to real bowling biomechanics
- **Prioritize user safety** - incorrect posture analysis could cause injury
- **Focus on educational value** - help users understand their technique

### When Debugging Issues
- **Start with MediaPipe** - most issues relate to pose detection
- **Check video quality** - poor videos lead to inaccurate analysis
- **Validate analysis logic** - ensure calculations match bowling science
- **Test with real bowling videos** - synthetic data doesn't represent real scenarios

Remember: Your ultimate goal is helping bowlers improve their technique safely and effectively. Every technical decision should support this objective.