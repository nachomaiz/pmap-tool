mkdir -p ~/.streamlit/

echo "[theme]
base = 'dark'
[server]
headless = true
port = $PORT
enableCORS = false
[browser]
gatherUsageStats = false
" > ~/.streamlit/config.toml