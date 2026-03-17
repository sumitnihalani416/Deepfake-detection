# Setup Guide for Deepfake Detection

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/sumitnihalani416/Deepfake-detection.git
   cd Deepfake-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dependency Setup
- Python 3.6 or higher
- Flask for the web application
- TensorFlow or PyTorch for model inference
- Other dependencies as listed in `requirements.txt`

## Running the Web Application
1. Navigate to the project directory:
   ```bash
   cd Deepfake-detection
   ```

2. Start the Flask application:
   ```bash
   python app.py
   ```

3. Open your browser and go to `http://localhost:5000` to access the application.

## API Usage Examples
- **POST /upload**: Upload a video for detection.
  - Request Body: 
    - `file`: video file (supported formats: mp4, avi)
  - Example:
    ```bash
    curl -X POST -F 'file=@video.mp4' http://localhost:5000/upload
    ```

- **GET /status**: Check the status of the detection process.
  - Example:
    ```bash
    curl http://localhost:5000/status
    ```

## Testing Procedures
1. Unit tests are contained within the `tests` directory.
2. Run tests using:
   ```bash
   pytest
   ```

## Troubleshooting
- If you encounter errors while running the app, check the following:
  - Ensure all dependencies are installed.
  - Verify your Python version is compatible.
  - Check the logs for error messages.