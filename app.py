import streamlit as st
import pandas as pd
import joblib



# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    return model

# Main app
def main():
    st.set_page_config(page_title="Student Performance Predictor", page_icon="📚", layout="wide")
    
    st.title("📚 Student Performance Predictor")
    st.markdown("---")
    st.write("This app predicts student performance index based on various study-related factors.")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📊 Input Features")
        
        # Input fields
        hours_studied = st.number_input(
            "Hours Studied per day",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5,
            format="%.1f",
            help="Number of hours spent studying per day"
        )
        
        previous_scores = st.slider(
            "Previous Scores",
            min_value=0,
            max_value=100,
            value=75,
            help="Previous academic scores (0-100)"
        )
        
        extracurricular = st.selectbox(
            "Extracurricular Activities",
            options=["Yes", "No"],
            help="Participation in extracurricular activities"
        )
        
        sleep_hours = st.number_input(
            "Sleep Hours per day",
            min_value=0.0,
            max_value=24.0,
            value=8.0,
            step=0.5,
            format="%.1f",
            help="Number of hours of sleep per day"
        )
        
        sample_papers = st.number_input(
            "Sample Question Papers Practiced",
            min_value=0,
            value=5,
            step=1,
            help="Number of sample question papers practiced"
        )
    
    with col2:
        st.header("🎯 Prediction")
        
        # Load model and make prediction
        try:
            model = load_model()
            
            # Prepare input data
            extracurricular_encoded = 1 if extracurricular == "Yes" else 0
            
            input_data = pd.DataFrame({
                'Hours Studied': [hours_studied],
                'Previous Scores': [previous_scores],
                'Extracurricular Activities': [extracurricular_encoded],
                'Sleep Hours': [sleep_hours],
                'Sample Question Papers Practiced': [sample_papers]
            })
            
            # Make prediction
            if st.button("🔮 Predict Performance", type="primary"):
                prediction = model.predict(input_data)[0]
                
                # Display result
                st.success(f"**Predicted Performance Index: {prediction:.2f}**")
                
                # Add performance interpretation
                if prediction >= 80:
                    st.balloons()
                    st.markdown("🌟 **Excellent Performance!** Keep up the great work!")
                elif prediction >= 60:
                    st.markdown("👍 **Good Performance!** You're doing well!")
                elif prediction >= 40:
                    st.markdown("⚠️ **Average Performance.** Consider studying more or getting better sleep.")
                else:
                    st.markdown("📈 **Below Average Performance.** Focus on improving study habits.")
                
                # Display input summary
                st.markdown("---")
                st.subheader("📋 Input Summary")
                st.write(f"• **Hours Studied:** {hours_studied} hours/day")
                st.write(f"• **Previous Scores:** {previous_scores}%")
                st.write(f"• **Extracurricular Activities:** {extracurricular}")
                st.write(f"• **Sleep Hours:** {sleep_hours} hours/day")
                st.write(f"• **Sample Papers Practiced:** {sample_papers}")
                
        except FileNotFoundError:
            st.error("❌ Model file 'model.pkl' not found. Please make sure the model is trained and saved.")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
    
    # Additional information
    st.markdown("---")
    st.markdown("### 💡 Tips for Better Performance")
    st.markdown("""
    - **Study consistently:** Maintain regular study hours
    - **Get adequate sleep:** 7-9 hours of sleep is optimal for learning
    - **Practice more:** Solve sample papers to improve performance
    - **Stay active:** Participate in extracurricular activities for balanced development
    - **Monitor progress:** Keep track of your scores and improvement areas
    """)

if __name__ == "__main__":
    main() 
