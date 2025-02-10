import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
import google.generativeai as genai

# Configuration for Gemini API
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "AIzaSyA3gDijvX9jV_DDv64QoNg1jCkIg8I6PZU")
genai.configure(api_key=GOOGLE_API_KEY)

class AirCanvasApp:
    def __init__(self):
        # MediaPipe hand tracking setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.7
        )
        
        # Drawing configuration
        self.drawing = False
        self.last_x, self.last_y = None, None
        self.canvas = None
        
    def create_drawing_canvas(self, frame):
        # Create a canvas for drawing
        self.canvas = np.zeros_like(frame)
        self.canvas.fill(255)  # White background
        return self.canvas
    
    def process_hand_tracking(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand tracking
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get index finger tip coordinates
                h, w, _ = frame.shape
                x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
                y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
                
                # Drawing logic
                if self.drawing:
                    cv2.line(self.canvas, 
                             (self.last_x, self.last_y), 
                             (x, y), 
                             (0, 0, 0), 
                             thickness=2)
                
                self.last_x, self.last_y = x, y
                self.drawing = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < \
                               hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y
        
        return self.canvas

def main():
    st.set_page_config(
        page_title="AI Air Canvas",
        page_icon="🎨",
        layout="wide"
    )
    
    # Custom CSS for professional look
    st.markdown("""
    <style>
    .main-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("🎨 AI Air Canvas & Chat")
    
    # Instantiate Air Canvas
    air_canvas_app = AirCanvasApp()
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Drawing Canvas")
        # Video capture
        cap = cv2.VideoCapture(0)
        
        # Streamlit image placeholder
        canvas_placeholder = st.empty()
        
        if cap.isOpened():
            ret, frame = cap.read()
            
            # Flip frame horizontally for natural drawing experience
            frame = cv2.flip(frame, 1)
            
            # Create drawing canvas
            canvas = air_canvas_app.create_drawing_canvas(frame)
            
            # Process hand tracking and drawing
            while ret:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                
                # Draw on canvas
                canvas = air_canvas_app.process_hand_tracking(frame)
                
                # Combine frame and canvas
                output = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
                
                # Display in Streamlit
                canvas_placeholder.image(output, channels="BGR")
                
                # Break loop if window is closed
                if not ret:
                    break
            
            cap.release()
        
        # Save Canvas Button
        if st.button("Save Drawing"):
            temp_file = tempfile.mktemp(suffix=".png")
            cv2.imwrite(temp_file, canvas)
            st.session_state['drawing_path'] = temp_file
            st.success("Drawing saved successfully!")
    
    with col2:
        st.subheader("Gemini Chat Interface")
        
        # Chat interface
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Enter your prompt"):
            # User message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Check if drawing exists
            if 'drawing_path' in st.session_state:
                # Initialize Gemini model
                model = genai.GenerativeModel('gemini-1.5-pro')
                
                # Read image
                with open(st.session_state['drawing_path'], "rb") as img_file:
                    img_data = img_file.read()
                
                # Generate response
                response = model.generate_content([prompt, img_data])
                
                # Bot message
                bot_response = response.text
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
                with st.chat_message("assistant"):
                    st.markdown(bot_response)
            else:
                st.warning("Please save a drawing first!")

if __name__ == "__main__":
    main()