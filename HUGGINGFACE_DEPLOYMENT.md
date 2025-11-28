# 🚀 Hugging Face Spaces Deployment Guide

## Prerequisites

1. **Hugging Face Account**: Sign up at https://huggingface.co/join
2. **Git LFS**: Install Git Large File Storage for handling large model files
   ```bash
   # Windows (using Git for Windows - already included)
   # Or download from: https://git-lfs.github.com/
   
   # Verify installation
   git lfs version
   ```

## 📋 Step-by-Step Deployment

### Step 1: Create a New Space on Hugging Face

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in the details:
   - **Space name**: `sign-language-recognizer` (or your preferred name)
   - **License**: MIT
   - **Select the SDK**: Choose **Gradio**
   - **Space hardware**: CPU (free tier) or upgrade to GPU if needed
   - **Visibility**: Public or Private

4. Click **"Create Space"**

### Step 2: Clone Your New Space Repository

```powershell
# Navigate to a folder where you want to clone
cd "D:\MS academic Projects\SD for AI\phase1"

# Clone the space (replace YOUR_USERNAME with your HF username)
git clone https://huggingface.co/spaces/YOUR_USERNAME/sign-language-recognizer
cd sign-language-recognizer

# Initialize Git LFS
git lfs install
git lfs track "*.pb"
```

### Step 3: Copy Required Files from Your Project

Copy these files from your current project to the cloned space folder:

**Essential Files:**
```powershell
# From your project root
$source = "d:\MS academic Projects\SD for AI\phase1\original\sign-language-alphabet-recognizer-master"
$dest = "D:\MS academic Projects\SD for AI\phase1\sign-language-recognizer"

# Copy main files
Copy-Item "$source\app_gradio.py" -Destination $dest
Copy-Item "$source\requirements.txt" -Destination $dest
Copy-Item "$source\.gitattributes" -Destination $dest

# Copy model files
Copy-Item "$source\logs\output_graph_improved.pb" -Destination "$dest\logs\"
Copy-Item "$source\logs\output_labels_improved.txt" -Destination "$dest\logs\"

# Copy sample dataset images (for examples in the UI)
Copy-Item "$source\dataset\A\1.jpg" -Destination "$dest\dataset\A\" -Force
Copy-Item "$source\dataset\B\1.jpg" -Destination "$dest\dataset\B\" -Force
Copy-Item "$source\dataset\C\1.jpg" -Destination "$dest\dataset\C\" -Force
```

### Step 4: Create README.md for Hugging Face

Create a `README.md` file in your space folder with this content:

```markdown
---
title: Sign Language Alphabet Recognizer
emoji: 🤟
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app_gradio.py
pinned: false
license: mit
---

# 🤟 Sign Language Alphabet Recognizer

A machine learning application that recognizes American Sign Language (ASL) alphabet gestures using InceptionV3 transfer learning.

## 🚀 Features
- Real-time recognition via webcam or image upload
- 29 classes: A-Z + Space + Del + Nothing
- Improved model with 10,000 training steps
- Confidence scores for predictions

## 📊 Model Details
- Framework: TensorFlow 2.11.0
- Architecture: InceptionV3
- Training Steps: 10,000
- Model Size: ~84 MB

## 👥 Team - Group 4
Fayaz Shaik, Harsha Koritala, Mallikarjun Kotha, Sai Grishyanth Magunta, Sai Kiran Dasari

[GitHub Repository](https://github.com/ShaikFayaz6/sign-language-recognizer)
```

### Step 5: Push to Hugging Face

```powershell
# Add all files
git add .

# Commit
git commit -m "Initial deployment: Sign Language Recognizer with improved model"

# Push to Hugging Face (may take time due to large model file)
git push
```

### Step 6: Wait for Build

1. Go to your Space on Hugging Face: `https://huggingface.co/spaces/YOUR_USERNAME/sign-language-recognizer`
2. The space will automatically build (takes 2-5 minutes)
3. Once built, your app will be live!

## 🎯 Testing Your Deployed App

1. Visit your space URL
2. Upload a sign language image or use webcam
3. View predictions with confidence scores

## 🔧 Troubleshooting

### Issue: Model file too large for Git
**Solution**: Git LFS is handling it, but if issues persist:
```powershell
git lfs migrate import --include="*.pb"
```

### Issue: Dependencies not installing
**Solution**: Check that `requirements.txt` has all dependencies:
- tensorflow==2.11.0
- gradio>=4.0.0
- opencv-python-headless
- numpy
- Pillow

### Issue: App crashes on startup
**Solution**: Check logs in Hugging Face Space settings, ensure:
- Model files exist in `logs/` folder
- File paths in `app_gradio.py` are correct
- All dependencies are installed

## 📱 Share Your Space

Once deployed, share your space:
- **Direct Link**: `https://huggingface.co/spaces/YOUR_USERNAME/sign-language-recognizer`
- **Embed in Website**: Use the embed code from HF Space settings
- **API Access**: Available via Gradio Client

## 🎉 Next Steps

- Add more example images
- Improve UI with custom CSS
- Add performance metrics display
- Enable GPU for faster inference
- Set up automatic model updates

---

💡 **Pro Tip**: You can update your app anytime by pushing changes to the Hugging Face repository!
