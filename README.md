---
title: DeepScan Video Detector
emoji: 🎥
colorFrom: slate
colorTo: emerald
sdk: gradio
app_file: app.py
pinned: false
---

# DeepScan AI: 4-Layer Video Detector

An intelligent, multi-layered video deepfake and AI generation detector. The tool is styled as a retro-modern monospace developer terminal console, enabling you to inspect frame-by-frame mathematical anomalies, run local deep learning models, and fallback to advanced commercial APIs—all while keeping hosting and execution cost-free.

---

## 🏗️ Core Architecture & Features
1. **Layer 1: Temporal Motion Flow & SSIM Math (Local)**: Computes frame-to-frame pixel displacements (dense optical flow) and structural similarity index (SSIM) over 12 uniformly sampled frames. Automatically adjusts to the video’s resolution and action levels to reduce false alarms.
2. **Layer 2: Adaptive Frequency Domain FFT (Local)**: Measures high-frequency noise loss patterns typical of GAN and Diffusion models. Resolution-independent and normalized to resist compression artifacts.
3. **Layer 3: Local Face Crop Classifier (Local)**: Employs a face tracker (**MediaPipe**) to extract faces, passing crops to a pre-trained **EfficientNet-B4** model. If pre-trained weights cannot be loaded locally, a fallback visual edge-frequency classification module is activated.
4. **Layer 4: Sightengine Premium API (Optional)**: Leverages Sightengine's enterprise video analysis backend for state-of-the-art text-to-video generators. Uses backend credentials so users do not need to provide their own keys.
5. **Daily Quotas & User Accounts**: A local SQLite database manages accounts with strict rate-limiting:
   - Max **5 total scans per day** per account.
   - Max **2 premium API scans per day** per account.
6. **Video Length Boundary**: Rejects videos longer than **12.0 seconds** immediately without costing quota.
7. **Automated API Fallback**: If the API call fails or the developer key is exhausted, it prints a warning to the terminal logs and shifts to local models automatically to complete the scan.
8. **Adaptive UI Locking**: Once the user hits their 2 premium scans limit, the premium API checkbox is automatically disabled and Standard check is locked on.

---

## 🚀 Installation & Local Run

### 1. Install Dependencies
Make sure you have Python 3.8+ installed. Install the requirements:
```bash
pip install -r requirements.txt
```

### 2. Configure Sightengine API Credentials (Optional)
To enable Layer 4 (Sightengine Premium), export your developer credentials as environment variables:

On macOS/Linux:
```bash
export SIGHTENGINE_API_USER="your_api_user_id"
export SIGHTENGINE_API_SECRET="your_api_secret_key"
```

On Windows (Command Prompt):
```cmd
set SIGHTENGINE_API_USER=your_api_user_id
set SIGHTENGINE_API_SECRET=your_api_secret_key
```

On Windows (PowerShell):
```powershell
$env:SIGHTENGINE_API_USER="your_api_user_id"
$env:SIGHTENGINE_API_SECRET="your_api_secret_key"
```

*Note: If these variables are not set, the app will execute the Standard pipeline. Selecting the Premium checkbox without setting these variables will output a credentials warning and fallback to local models.*

### 3. Launch App
Run the web application locally:
```bash
python app.py
```
Open the printed URL (usually `http://127.0.0.1:7860`) in your browser to access the portal.

---

## 💾 User Database
User sessions and scans are tracked in `users.db` inside the project folder. To reset all operator accounts and quotas, simply delete `users.db` and restart the application.
