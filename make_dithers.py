#Program to make nearly optimal dithers of blended acquisition.
#Thomas Elboth - March/April 2021
#The general idea: We want to make rffective dither sequences that are:
# 1. nearly uniform random
# 2. a bit low discrepancy (anti clustering)


import streamlit as st
import numpy as np
import math
import random
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from statsmodels.tools import sequences #to get Halton (and similar models)
from scipy.stats import poisson

import os
import json
import pickle
import uuid
import re
import pandas as pd

def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.

    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.

    Returns:
    -------
    (str): the anchor tag to download object_to_download

    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')

    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f"""
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;

            }}
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link

#make random numbers in the 0-1 range - quickly
def make_random_numbers(numb):
    my_rand= np.zeros((numb))
    for i in range(0, numb):
        my_rand[i] = random.random()
    return my_rand

#a somewhat differnt poisson distribution - for testing
def makePoissonDist(size0):
    r = poisson.rvs(mu=9999, size=size0)
    r=r-min(r)
    r=r/max(r) #scale to [0-1] range
    r2=[]
    for i in range(0,size0):
        if(r[i]>0.4 and r[i]<0.6):
            r2.append(r[i])
    count=0
    while (len(r2)<size0):
        r2.append(r2[count])
        count=count+1
    r2=r2-min(r2)
    r2=r2/max(r2) #scale to [0-1] range
    r2=r2[0:size0]

    return r2


def plot_dithers(src_d, nLevels, dither_type):

    no_src = len(src_d)
    my_color=['red','green','blue','cyan','yellow','magneta','black','gray','brown','orange']
    if no_src<200:
        marker_size=3
    elif no_src<400:
        marker_size=2
    elif no_src<500:
        marker_size=1
    else:
        marker_size=0.5
    no_dithers = len(src_d.T)

    if (no_src==1 and nLevels==1): #One source dither - low discrepancy - uniform random....
        fig = make_subplots(rows=no_src, cols=2,
            subplot_titles=('Dither on S1', 'Histogram of S1')
        )
        fig.add_trace(
            go.Scatter(x=np.linspace(0,no_dithers,no_dithers), y=src_d[0,:], mode='markers', marker_size=marker_size,marker_color=my_color[0]),
            row=1, col=1 )
        fig.add_trace(
            go.Histogram(x=src_d[0,:], marker_color=my_color[0], nbins=20),
            row=1, col=2 )

    elif (no_src>1 and nLevels==1):  #multi-source - easing off on the uniform randomness to avoid potential patent issue
        list_of_titles=[] #create all the sub-plot text
        for i in range(0,no_src-1):
            list_of_titles.append('Dither on S'+str(i+1))
            list_of_titles.append('Dither on S'+str(i+1)+'(i)-S'+str(i+2)+'(i)')
            list_of_titles.append('Histogram of S'+str(i+1)+'(i)-S'+str(i+2)+'(i)')
        list_of_titles.append('Dither on S'+str(no_src))
        list_of_titles.append('Dither on S'+str(no_src)+'(i)-S'+str(1)+'(i+1)')
        list_of_titles.append('Histogram of S'+str(no_src)+'(i)-S'+str(1)+'(i+1)')

        fig = make_subplots(rows=no_src, cols=3,subplot_titles=(list_of_titles) )

    elif (nLevels==2 and no_src>2): #for 2 levels we only plot the data for 3 sources
            fig = make_subplots(rows=2, cols=3, subplot_titles=('Dither on S1','S1(i)-S2(i)','S1(i)-S3(i)'))
    else:
        print('Only up to 9 sources supported - for now...With nLevels==2, pick 3 or more sources')

    #precompute the effective dithers: (S1+S2, S2+S3 and S3+S1(i+1))
    e_dithers = np.zeros((no_src, no_dithers))
    for i in range(0, no_src-1):
        e_dithers[i,:] = src_d[i,:]-src_d[i+1,:]
    e_dithers[no_src-1,0:-2]= src_d[no_src-1,0:-2]-src_d[0,1:-1]

    if(nLevels==1):
        for i in range(0, no_src):

            fig.add_trace( #first row is the actual dithers on each source
                go.Scatter(x=np.linspace(0,no_dithers,no_dithers), y=src_d[i,:], mode='markers', marker_size=marker_size,marker_color=my_color[i]),
                row=i+1, col=1)

            fig.add_trace( #second row is the effective dithers (S1+S2, S2+S3 and S3+S1(i+1))
                go.Scatter(x=np.linspace(0,no_dithers,no_dithers), y=e_dithers[i,:], mode='markers', marker_size=marker_size,marker_color=my_color[i]),
                row=i+1, col=2)

            fig.add_trace( #the last row is the histograms of the effective dither sequence
                go.Histogram(x=e_dithers[i,:], marker_color=my_color[i]),
                row=i+1, col=3)
            txt=dither_type+" dither sequence for blended acquisition."
            fig.update_layout(title_text=txt)

    elif(nLevels==2):
        e_dithers2 = np.zeros((no_src, no_dithers))
        for i in range(0, no_src-2):
            e_dithers2[i,:] = src_d[i,:]-src_d[i+2,:]
        e_dithers2[no_src-2,0:-2]= src_d[no_src-2,0:-2]-src_d[0,1:-1]
        e_dithers2[no_src-1,0:-2]= src_d[no_src-1,0:-2]-src_d[1,1:-1]

        fig.add_trace(     #first row is the actual dithers on each source
            go.Scatter(x=np.linspace(0,no_dithers,no_dithers), y=src_d[0,:], mode='markers', marker_size=marker_size,marker_color=my_color[0]),
            row=1, col=1)
        fig.add_trace(    #second row is the effective dithers (S1+S2, S2+S3 and S3+S1(i+1))
            go.Scatter(x=np.linspace(0,no_dithers,no_dithers), y=e_dithers[0,:], mode='markers', marker_size=marker_size,marker_color=my_color[0]),
            row=1, col=2)
        fig.add_trace(    #the last row is the histograms of the effective dither sequence
            go.Scatter(x=np.linspace(0,no_dithers,no_dithers), y=e_dithers2[0,:], mode='markers', marker_size=marker_size,marker_color=my_color[1]),
            row=1, col=3)
        fig.add_trace(
                go.Histogram(x=e_dithers[0,:], marker_color=my_color[0]),
                row=2, col=2)
        fig.add_trace(
                go.Histogram(x=e_dithers2[0,:], marker_color=my_color[1]),
                row=2, col=3)
        fig.update_layout(title_text="Dithering sequence optimized for both N+1 and N+2 shots, with histograms")

    fig.update_layout(height=600, width=800,  showlegend=False)
    return fig
    #fig.show()

def get_params(): #the streamlit stuff
    st.title('Dither sequences for blended acquisition')

    #start by getting the input parameters from the user. TODO: add tooltip
    my_expander1 = st.beta_expander("General parameters:", expanded=True)

    with my_expander1:
        col1, col2, col3 = st.beta_columns(3)
        with col1:
            no_src =        st.number_input('Number of souces to dither:',1,9,3,1,help="This is the number of sources operating in flip-flop-flap- mode.")
            nPoints =       st.number_input('The number of dithers per source:',10,2500,200,10, help='This is how many dithers you want for one source. Typically this number should be > no traces in the migration aperture.')
            compute_dithers=st.button("Plot the dithers (for QC)", help='Produce nice looking QC plots of the dither sequences.')
            get_help       =st.button("Get a ppt that explains the dithering",help="Download a ppt with a lot of explanation on why you want to use the inverse Irwin-Hall distribution.")
            if get_help: #return a ppt
                # Load selected file
                filename="./dither_explained.pptx"
                with open(filename, 'rb') as f:
                    s = f.read()
                    download_button_str = download_button(s, filename, f'Click here to download {filename}')
                    st.markdown(download_button_str, unsafe_allow_html=True)
        with col2:
            range_beg= st.number_input('Dither minimum in ms:', -2000,2000, 0, 4)
            range_end= st.number_input('Dither maximum in ms:', -2000,2000, 500, 4)
            dither_type=st.selectbox('Select type of dithers:',('Inverse Irwin-Hall','Random','Halton','Poisson'), help="The IHH (Invese Irwin-Hall) is the one your should select! The others are for RnD and as illustrations.")
        with col3:
            nLevels = st.number_input('Number of levels (N+1 or N+2):',1,2,1,1, help="Keep this as 1 for all normal surveys. In a case where sources are going off very often, it might be advisable to also optimize the dithers for the N+2 shot. Pls contact RnD before using this on a real survey.")
            nBacksteps = st.number_input('Amount of anti-clustering:',1,5,5,1,help="Keep this at 5 or less. [3-5] is a good and robust choice. If you use numbers much larger than 5, there is a potential issue with regards to a CGG patent in that the effective distribution becones close to uniform random. Please contact Legual/IP council and RnD before going above 5.")
            user_seed = int(st.text_input("User seed (for rand numb gen):", "0", help="Keeping this at 0 will give different results each time, However, by providing your own seed, for example 123, the same random sequence is produced every time."))

    if(user_seed!=0):
        random.seed(a=user_seed)

    return [nBacksteps, no_src, nPoints, range_beg, range_end, nLevels, dither_type, compute_dithers]


def make_dithers():

    [nBacksteps, no_src, nPoints, range_beg, range_end, nLevels, dither_type, compute_dithers] = get_params()

    if compute_dithers:
        if (dither_type=='Inverse Irwin-Hall'): #the inverse Irwin Hall distribution (DEFAIÙLT)
            f = np.zeros((nBacksteps))
            f1 =np.zeros((nBacksteps))
            Numb = np.zeros((no_src))

            src_d =   np.zeros((no_src, nPoints+nBacksteps+1+2))    #the array with the dither values
            no_trials = 2000							#important parameter for speed - TODO: look at speedy randum mumber computations

            if(nLevels==2 or nBacksteps >10):
                no_trials=no_trials*2  #a bit more testing is recquired in this case

            #The anti-clustering to produce inverse Irwin-Hall distributions
            for i in range(0, nBacksteps):
                if(no_src<=1):
                    f1[i] = 0.12*(0.90**i)  #make this number a parameter [~0.15]
                else:
                    f1[i] = 0.30*(0.90**i)  #make this number a parameter [0.25-0.35]
                f[i] = f1[i]

            t=nBacksteps+1
            count=0      #total number of trials in each iteration
            count2=0     #counter for the random number array (my_random)
            scale=0

            #it is efficient to produce many random numbers at a time
            my_random = make_random_numbers(no_trials*no_src) #make som initial random numbers

            while (t<nPoints+nBacksteps+2):

                #Gradually scale the f-weights down to be sure we always get a solution
                if (no_trials % (count+1)==0):
                    count2=0
                    tmp = 0.93**scale
                    f=f1*tmp
                    my_random =  make_random_numbers(no_trials*no_src)
                    scale=scale+1

                my_coutinue=True
                Numb = my_random[count2:count2+no_src] #Draw candidate dithering times

                if(no_src==1 and nLevels==1):
                    for i in range(0, nBacksteps):			#Does candidate fulfill anti-clustering recq?
                        if ( abs(my_random[count2]-src_d[0][t-i]) < 0.5*f[i] ):
                            my_coutinue=False; break;  #exit - no solution found

                elif(no_src>=2 and nLevels==1):            #General solution for any number of sources (1 level)
                    for i in range(0, nBacksteps):
                        for j in range(0, no_src-1):
                            if(abs((my_random[count2+j]-my_random[count2+j+1])-(src_d[j][t-i]-src_d[j+1][t-i])) <f[i]):
                                my_coutinue=False; break;  #S0(i)-S1(i),...,SN-1(i)-SN(i)

                        if(abs((src_d[no_src-1][t]-my_random[count2])-(src_d[no_src-1][t-i-1]-src_d[0][t-i])) <f[i]):
                            my_coutinue=False              #SN(i)-S0(i+1)


                elif(no_src>=2 and nLevels==2):            #General solution for any number of sources (2 level optimization)
                    for i in range(0, nBacksteps):
                        for j in range(0, no_src-1):
                            if(abs((my_random[count2+j]-my_random[count2+j+1])-(src_d[j][t-i]-src_d[j+1][t-i]))<f[i]):
                                my_coutinue=False; break;  #S0(i)-S1(i),...,SN-1(i)-SN(i)

                        if(abs((src_d[no_src-1][t]-my_random[count2])-(src_d[no_src-1][t-i-1]-src_d[0][t-i]))<f[i]):
                            my_coutinue=False              #SN(i)-S0(i+1)

                        for j in range(0, no_src-2):       #and here is N+2
                            if(abs((my_random[count2+j]-my_random[count2+j+2])-(src_d[j][t-i]-src_d[j+2][t-i])) <f[i]):
                                my_coutinue=False; break;  #S0(i)-S2(i),...,SN-2(i)-SN(i)

                        if(abs((src_d[no_src-2][t]-my_random[count2])-(src_d[no_src-2][t-i-1]-src_d[0][t-i]))<f[i]):
                            my_coutinue=False;             #SN-1(i)-S0(i+1)
                        if(abs((src_d[no_src-1][t]-my_random[count2+1])-(src_d[no_src-1][t-i-1]-src_d[1][t-i]))<f[i]):
                            my_coutinue=False               #SN(i)-S1(i+1)

                else:
                    print('Something is wrong....')

                if (my_coutinue == True):  #solution found, and accepted
                    t=t+1; count=0; count2=0; scale=0;
                    f=f1
                    src_d[:,t] = Numb
                    my_random =  make_random_numbers(no_trials*no_src) # !make new random numbers for next iteration

                count=count+1;        #keep track of the number of trials
                count2=count2+no_src; #counter for the random number

            #cut away the first few numbers:
            src_d = src_d[:,nBacksteps+2:-1]

        elif (dither_type=='Random'):
            src_d = np.zeros((no_src, nPoints))
            for i in range(0,no_src):
                for j in range(0, nPoints):
                    src_d[i,j] = random.random()

        elif (dither_type=='Halton'):
            src_d = np.zeros((no_src, nPoints))
            for i in range(0,no_src):
                src_d[i,:] = sequences.van_der_corput(nPoints, start_index=123+100*i)

        elif (dither_type=='Poisson'):
            src_d = np.zeros((no_src, nPoints))
            for i in range(0,no_src):
                src_d[i,:] = makePoissonDist(nPoints)

        else:
            print("Not implemented yet...") #we should not come here

        #scale the dither to be in the correct range
        src_d = np.round((src_d*abs(range_end-range_beg))+range_beg)

        #go through - making sure every dither calue is a multiople of 2ms
        for i in range(0,no_src):
            for j in range(0, nPoints):
                if (src_d[i,j] % 2 !=0):
                    src_d[i,j] = src_d[i,j] +1

        fig=plot_dithers(src_d, nLevels, dither_type)
        if (len(fig.data)>0):
            st.plotly_chart(fig,use_container_width=False)

        #to allow the user to download the file
        df = pd.DataFrame(src_d.T)
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        st.markdown(href, unsafe_allow_html=True)

        st.write("The computed dither sequence:")
        #st.write(src_d.T)
        #st.dataframe(df.style.highlight_max(axis=0))
        st.table(df)

def main():
    print("Hello World!")
    make_dithers()

if __name__ == "__main__":
    main()
