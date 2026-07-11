# EcoPlate AI

## Overview

EcoPlate AI is an AI-powered food waste detection system that estimates leftover food quantity from plate images using Machine Learning and Computer Vision.

## Features

- Upload food plate images
- AI waste prediction
- Waste percentage estimation
- Estimated leftover weight
- Prediction history
- Analytics dashboard
- CSV Export

## Technologies

### Frontend

- Next.js
- React
- Tailwind CSS

### Backend

- FastAPI
- Python
- OpenCV
- Scikit-Learn
- SQLite

## Installation

### Backend

```bash
cd ecoplate-backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd ecoplate-frontend
npm install
npm run dev
```

Frontend

```
http://localhost:3000
```

Backend

```
http://127.0.0.1:8000/docs
```