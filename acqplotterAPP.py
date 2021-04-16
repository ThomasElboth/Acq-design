#Python program to study/plot/compare the offset distrobution in node
#and streamer surveys
#Thomas Elboth - Shearwater geoservice Feb-Mar 2021

#pip install numpy streamlit matplotlib plotly

import streamlit as st
import numpy as np
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

#The varoius python scripts to include
import acq_plots as pl
import vib_directivity as v
import make_dithers as md



def main():
    #print("Hello World!")
    st.set_page_config(layout="wide")
    st.sidebar.image('Shearwater_logo400x85.png')
    st.sidebar.subheader('Marine seismic acquisition tools')
    st.sidebar.subheader('Version 0.1.7 April 13 - 2021')
    st.sidebar.write('Shearwater reflection marine RnD tools.')
    st.sidebar.write('The code is continiously under development, with new features added all the time. There are bugs in this code!')
    st.sidebar.write('If you have questions, comments or features you would like to see/add, please contact telboth@shearwatergeo.com')

    if(st.sidebar.checkbox('acq_design', value=True, help='Tick this to go to the acq_design sub-page')):
        pl.acq_plots()
    if(st.sidebar.checkbox('source_design', value=False, help='Tick this to go to the source_design sub-page, which is is work in progress...')):
        v.plot_array_directivity()
    if(st.sidebar.checkbox('make_dithers', value=False, help='Tick this to go to a page that helps produce dithering sequences for blended acquisition.')):
        md.make_dithers()


    st.sidebar.write('Pick one of the above')



if __name__ == "__main__":
    main()
