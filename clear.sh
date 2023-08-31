#!/bin/bash


PORT_FLASK=5000
PORT_STREAMLIT=8501


PID_FLASK=$(fuser $PORT_FLASK/tcp 2>/dev/null)
PID_STREAMLIT=$(fuser $PORT_STREAMLIT/tcp 2>/dev/null)
if [ -n "$PID_FLASK" ]; then
    kill $PID_FLASK
fi

if [ -n "$PID_STREAMLIT" ]; then
    kill $PID_STREAMLIT
fi
