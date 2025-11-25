# Deploy Sign Language Recognizer to Free Hosting Platforms

This guide explains how to deploy your Sign Language Alphabet Recognizer to various free hosting platforms.

## 🚀 Deployment Options

### Option 1: Render.com (Recommended) ⭐

**Free Tier:** 750 hours/month, auto-sleep after 15 min inactivity

**Steps:**
1. Push your code to GitHub
2. Go to [render.com](https://render.com) and sign up
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Render will auto-detect `render.yaml` and configure everything
6. Click "Create Web Service"
7. Wait 5-10 minutes for deployment
8. Your app will be live at `https://your-app-name.onrender.com`

**Pros:**
- Easiest deployment (one-click with render.yaml)
- Generous free tier (750 hours)
- Auto-deploys on git push
- Free SSL certificate

**Cons:**
- Cold start after 15 min inactivity (~30 seconds)
- Limited to 512MB RAM

---

### Option 2: Railway.app

**Free Tier:** $5 credit/month (enough for ~150-200 hours)

**Steps:**
1. Push code to GitHub
2. Go to [railway.app](https://railway.app) and sign up
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will detect `railway.json` and `Procfile`
6. Click "Deploy"
7. Get your public URL from the deployment settings

**Pros:**
- Fast deployment and cold starts
- Better performance than Render
- Modern UI/UX

**Cons:**
- Limited free tier ($5/month credit)
- May run out of credit mid-month

---

### Option 3: Heroku (Legacy Free Tier Removed)

**Note:** Heroku removed their free tier in November 2022. You need a paid plan ($5/month minimum).

**Steps (if you have paid plan):**
1. Install Heroku CLI: `npm install -g heroku`
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Push code: `git push heroku main`
5. Scale: `heroku ps:scale web=1`

---

### Option 4: Fly.io

**Free Tier:** 3 shared VMs, 3GB persistent storage

**Steps:**
1. Install flyctl: `powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"`
2. Sign up: `flyctl auth signup`
3. Launch app: `flyctl launch`
4. Select region and confirm
5. Deploy: `flyctl deploy`

**Pros:**
- Good free tier
- Better performance
- No auto-sleep

**Cons:**
- Requires CLI setup
- More complex configuration

---

### Option 5: Google Cloud Run (Free Tier)

**Free Tier:** 2 million requests/month, always free

**Steps:**
1. Install Google Cloud SDK
2. Build Docker image: `docker build -t gcr.io/PROJECT_ID/sign-language-recognizer -f Dockerfile.cloud .`
3. Push image: `docker push gcr.io/PROJECT_ID/sign-language-recognizer`
4. Deploy: `gcloud run deploy --image gcr.io/PROJECT_ID/sign-language-recognizer --platform managed`

**Pros:**
- Generous free tier
- Scales to zero (no cost when idle)
- Enterprise-grade

**Cons:**
- Requires Docker knowledge
- More setup complexity

---

## 📝 Pre-Deployment Checklist

Before deploying, make sure:

- ✅ `requirements.txt` includes `gunicorn` and `opencv-python-headless`
- ✅ Model files are in `logs/` directory (~88MB)
- ✅ Database file `predictions.db` will be created automatically
- ✅ Environment variables are set (if needed)
- ✅ Git repository is up to date

## 🔧 Configuration Files Created

- **`render.yaml`** - Render.com configuration
- **`railway.json`** - Railway.app configuration  
- **`Procfile`** - Process file for Heroku/Railway
- **`runtime.txt`** - Python version specification
- **`Dockerfile.cloud`** - Docker configuration for containerized deployment

## ⚠️ Important Notes

### Model File Size
Your model file (`logs/output_graph.pb`) is ~88MB. Most platforms allow this, but:
- **Render/Railway:** No issues
- **Heroku:** Max slug size is 500MB (you're fine)
- **GitHub:** Use Git LFS for files >100MB

### Database Persistence
SQLite database (`predictions.db`) will be stored in the container. On free tiers:
- **Render/Railway:** Database resets on container restart
- **Solution:** Use free PostgreSQL add-on for persistence (recommended for production)

### Cold Starts
Free tier apps sleep after inactivity:
- **Render:** 15 minutes → 30-second cold start
- **Railway:** No auto-sleep (until credit runs out)
- **Fly.io:** No auto-sleep

### Performance Optimization
For better performance on free tiers:
1. Model loads once at startup (~5-10 seconds)
2. Use `gunicorn` with 2 workers
3. Set timeout to 120 seconds for model loading
4. Use `opencv-python-headless` (smaller package)

## 🚦 Recommended Deployment Path

**For beginners:**
1. Start with **Render.com** (easiest, most reliable)
2. Push to GitHub
3. Connect to Render
4. Deploy with one click

**For better performance:**
1. Use **Railway.app** (faster, no sleep)
2. Monitor credit usage
3. Upgrade to paid if needed ($5/month)

## 🔗 Post-Deployment

After deployment:
1. Test all routes: `/`, `/upload`, `/webcam`, `/history`
2. Upload a test image to verify model works
3. Check webcam functionality (may not work on some browsers)
4. Monitor logs for errors
5. Share your deployed URL! 🎉

## 📊 Expected Performance

- **Model Load Time:** 5-10 seconds (first request)
- **Prediction Time:** <2 seconds per image
- **Cold Start:** 30-60 seconds (after sleep)
- **Memory Usage:** ~500MB (model in memory)

## 🆘 Troubleshooting

**App won't start:**
- Check logs: `render logs` or platform dashboard
- Verify all files are committed to Git
- Ensure `logs/output_graph.pb` exists

**Out of memory:**
- Use `gunicorn --workers 1` (instead of 2)
- Upgrade to paid tier for more RAM

**Model not loading:**
- Check file paths are correct
- Verify model files are in Git repository
- Check Git LFS if file >100MB

**Database resets:**
- Use PostgreSQL add-on (free on Render/Railway)
- Update `database.py` to use PostgreSQL

---

## 🎯 Quick Start (Render.com)

```bash
# 1. Commit all files
git add .
git commit -m "Add deployment configuration"
git push origin main

# 2. Go to render.com and sign up

# 3. Click "New +" → "Web Service"

# 4. Connect GitHub repo

# 5. Render auto-detects render.yaml

# 6. Click "Create Web Service"

# 7. Wait 5-10 minutes

# 8. Visit your deployed app! 🚀
```

Your app will be live at: `https://your-app-name.onrender.com`

---

**Need help?** Check the platform-specific documentation:
- [Render Docs](https://render.com/docs)
- [Railway Docs](https://docs.railway.app)
- [Fly.io Docs](https://fly.io/docs)
