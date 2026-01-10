# Password Assistant - Beautiful ML-Powered Password Analyzer

A stunning Flask web application that analyzes password strength with vibrant gradients, smooth animations, and an intuitive interface.

## Features

- **Beautiful Design**: Vibrant gradient backgrounds with animated floating orbs
- **Login-Style Interface**: Circular animated card with spinning rings
- **Full-Screen Results**: Password analysis opens in a new window with large, readable fonts
- **ML-Powered Analysis**: Uses enhanced password strength model
- **Comprehensive Feedback**: Shows strength, crack time estimates, issues, and suggestions

## Running the Application

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server**:
   ```bash
   cd password-ml-app
   PORT=5555 python app.py
   ```

3. **Access the App**:
   - In Codespaces: Forward port 5555 in the "Ports" panel
   - Open the forwarded URL in your browser
   - Or visit `http://localhost:5555` if running locally

## Design Highlights

### Login Page
- **Vibrant gradient background**: Purple to pink to blue (`#667eea → #764ba2 → #f093fb`)
- **Animated floating orbs**: Smooth, continuous movement in the background
- **Circular login card**: White card with pulsing shadow animation
- **Spinning rings**: Three animated rings rotating around the card
- **Black text on white**: High contrast for easy readability

### Result Page
- **Large, bold fonts**: 
  - Title: 48px with gradient color
  - Strength label: 72px with pulsing animation
  - Body text: 18-20px for readability
- **Emoji icons**: 🔐 for security, 🕒 for time
- **Gradient backgrounds**: Smooth color transitions
- **Entrance animations**: Smooth scale and fade-in effects
- **Pulsing strength badge**: Eye-catching animated indicator

## API Endpoint

**POST** `/api/analyze`

Request body:
```json
{
  "password": "your_password_here"
}
```

Response:
```json
{
  "input_password": "your_password_here",
  "predicted_strength": "Strong",
  "estimated_crack_time": "centuries",
  "issues_found": [],
  "suggestions": []
}
```

## File Structure

```
password-ml-app/
├── app.py                      # Flask server
├── requirements.txt            # Python dependencies
├── engine/
│   └── ml_engine.py           # ML model and analysis engine
├── models/                     # ML model files
├── static/
│   └── style.css              # Vibrant styles and animations
└── templates_/
    └── index.html             # Main UI with embedded result page
```

## Technologies

- **Backend**: Flask, scikit-learn, joblib
- **Frontend**: HTML5, CSS3 (gradients, animations), Vanilla JavaScript
- **Design**: Gradient backgrounds, glassmorphism, keyframe animations
