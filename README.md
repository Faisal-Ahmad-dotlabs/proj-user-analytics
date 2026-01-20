# Ella Mixpanel Analytics Dashboard

A comprehensive analytics dashboard for tracking user behavior, content engagement, and session analytics from Mixpanel data.

## Features

### User Analytics
- User overview with sessions, active days, and last seen
- User attributes (location, device, languages)
- Individual user deep-dive analysis

### Content Analytics
- Content library overview with progress tracking
- Playback time and completion metrics
- Saved words tracking per content
- Interactive content history charts

### Session Analytics
- Daily session timeline visualization
- Detailed session breakdown with events
- Content playback within sessions

## Installation

### With Python (locally)

1. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

2. **Run the dashboard:**
  ```bash
  cp env.sample .env  # fill the secrets in the `.env` file
  python -m dashboard.main
  ```

3. **Access the dashboard:**
  Open the link `http://localhost:8000` in your browser.
  
### With Docker (locally)

1. **Build docker container:**
  ```bash
  docker build -t ella-analytics .
  ```

2. **Run the dashboard:**
  ```bash
  cp env.sample .env  # fill the secrets in the `.env` file
  docker run --env-file .env -p 8000:8000 -n ella-analytics ella-analytics
  ```

3. **Access the dashboard:**
  Open the link `http://localhost:8000` in your browser.
 
## Usage

1. **Fetch Data:** Select environment (Dev/Prod), choose date range, click "Fetch data"
2. **Select User:** Choose "All users" or select a specific user by `distinct_id`
3. **Explore Analytics:** View session timeline, content library, and detailed metrics
4. **Click for Details:** Click on content rows or timeline days for detailed views

## Configuration

API credentials are configured in the code:
- Mixpanel Dev/Prod secrets
- Content API endpoint and token

## Mobile Responsive

The dashboard is fully mobile-responsive with touch-optimized controls.

## Tech Stack

- **Dash** - Web framework
- **Plotly** - Interactive visualizations
- **Pandas** - Data processing
- **ngrok** - Public URL tunneling
