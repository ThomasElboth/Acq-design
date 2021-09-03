
import math
import numpy as np
import plotly.graph_objects as go
import streamlit as st

phase_add=st.slider("Phase delay added to the sweep (deg)", 0,360, 0,1)
phase_add=np.deg2rad(phase_add)      #to keep track of the phase

x=np.linspace(phase_add, phase_add+4*math.pi, 100)
y=np.sin(x)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines' ))
fig.update_layout(title="with plotly",  xaxis_title='x',yaxis_title='y')
st.plotly_chart(fig)
